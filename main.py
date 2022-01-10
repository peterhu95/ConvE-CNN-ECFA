import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import math

from os.path import join
import torch.backends.cudnn as cudnn

from evaluation import ranking_and_hits
from model import ConvE, DistMult, Complex, MyModel, ConvE2

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
import argparse

from copy import deepcopy
import datetime

np.set_printoptions(precision=3)

cudnn.benchmark = True


''' Preprocess knowledge graph using spodernet. '''
def preprocess(dataset_name, delete_data=False):
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(args.data, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()


    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)


def main(args, model1_path, model2_path):
    if args.preprocess: preprocess(args.data, delete_data=True)
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(args.data, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']        

    num_entities = vocab['e1'].num_token
    #print(vocab['e1'].token2idx) 
    #print(vocab['rel'].token2idx)        

    train_batcher = StreamBatcher(args.data, 'train', args.batch_size, randomize=True, keys=input_keys, loader_threads=args.loader_threads)
    dev_rank_batcher = StreamBatcher(args.data, 'dev_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)
    test_rank_batcher = StreamBatcher(args.data, 'test_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)


    if args.model is None:
        model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'conve':
        model1 = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
        model2 = ConvE2(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'distmult':
        model = DistMult(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'complex':
        model = Complex(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'mymodel':
        model = MyModel(args, vocab['e1'].num_token, vocab['rel'].num_token)
    else:
        log.info('Unknown model: {0}', args.model)
        raise Exception("Unknown model!")

    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))


    eta = ETAHook('train', print_every_x_batches=args.log_interval)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=args.log_interval))

    model1.cuda()
    model2.cuda()
    if args.resume:
        model1_params = torch.load(model1_path)
        model2_params = torch.load(model2_path)
        print(model1)
        print(model2)
        
        params1 = [(key, value.size(), value.numel()) for key, value in model1_params.items()]
        params2 = [(key, value.size(), value.numel()) for key, value in model2_params.items()]
        total_param_size = []
        for key, size, count in params1:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        
        total_param_size = []
        for key, size, count in params2:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        
        model1.load_state_dict(model1_params)
        model1.eval()
        model2.load_state_dict(model2_params)
        model2.eval()
        ranking_and_hits(model1, model2, test_rank_batcher, vocab, 'test_evaluation')
        ranking_and_hits(model1, model2, dev_rank_batcher, vocab, 'dev_evaluation')
    else:
        model1.init()
        model2.init()

    total_param_size = []
    params = [value.numel() for value in model1.parameters()]
    print(params)
    print(np.sum(params))
    total_param_size = []
    params = [value.numel() for value in model2.parameters()]
    print(params)
    print(np.sum(params))
   
    torch.autograd.set_detect_anomaly(True)
    opt1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.l2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.l2)
    
    best_valid_MRR = 0
    best_valid_H1 = 0
    best_valid_H3 = 0
    best_valid_H10 = 0
    best_test_MRR = 0
    best_test_H1 = 0
    best_test_H3 = 0
    best_test_H10 = 0
    
    patience = 150
    not_improve_epochs = 0
    lp=0
    for epoch in range(1, args.epochs+1):
        model1.train()
        model2.train()
        for i, str2var in enumerate(train_batcher):
            opt1.zero_grad()
            opt2.zero_grad()
            e1 = str2var['e1']
            rel = str2var['rel']
            e2_multi = str2var['e2_multi1_binary'].float()
            _, e2_idx = torch.nonzero(e2_multi, as_tuple=True)
            e2_idx = torch.LongTensor(e2_idx.cpu())
            # label smoothing
            e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.size(1))

            p = epoch / float(args.epochs)
            lp = 2./ (1. + np.exp(-10*p)) - 1
            
            pred1 = model1.forward(e1, rel, lp)
            pred2 = model2.forward(e1, rel, model1.rm_fea.detach())  # , model1.rm_emb.detach(), e2_idx
            loss1 = model1.loss(pred1*pred2, e2_multi)
            loss2 = model2.loss(pred2, e2_multi)
            
            loss1.backward(retain_graph=True)
            loss2.backward()
            opt1.step()
            opt2.step()
            
            train_batcher.state.loss = loss2.cpu()
        
        

        model1.eval()
        model2.eval()
        with torch.no_grad():
            if (epoch <= args.epochs/3 and epoch % 5 == 0) or (epoch > args.epochs/3 and epoch % 2 == 0):
                valid_MRR, valid_H1, valid_H3, valid_H10 = ranking_and_hits(model1, lp, model2, dev_rank_batcher, vocab, 'dev_evaluation')
                if (valid_MRR > best_valid_MRR and valid_H1 > best_valid_H1) or (valid_MRR > best_valid_MRR and valid_H3 > best_valid_H3):
                    best_valid_MRR = valid_MRR
                    best_valid_H1 = valid_H1
                    best_valid_H3 = valid_H3
                    best_valid_H10 = valid_H10
                    
                    best_test_MRR, best_test_H1, best_test_H3, best_test_H10 = ranking_and_hits(model1, lp, model2, test_rank_batcher, vocab, 'test_evaluation')
                    
                    not_improve_epochs = 0
                    # saving model1
                    print('saving to {0}'.format(model1_path))
                    torch.save(model1.state_dict(), model1_path)
                    # saving model2
                    print('saving to {0}'.format(model2_path))
                    torch.save(model2.state_dict(), model2_path)
                    
                else:
                    not_improve_epochs += 1
                
                print('\nnot_improve_epochs: {0}\tpatience: {1}'.format(not_improve_epochs, patience))
                print('\n'+ 'Current best valid MRR: '+str(best_valid_MRR))
                print('Current best valid Hits@1: '+str(best_valid_H1))
                print('Current best valid Hit@3: '+str(best_valid_H3))
                print('Current best valid Hit@10: '+str(best_valid_H10) +'\n')
                
                print('\n'+ 'Current best test MRR: '+str(best_test_MRR))
                print('Current best test Hits@1: '+str(best_test_H1))
                print('Current best test Hit@3: '+str(best_test_H3))
                print('Current best test Hit@10: '+str(best_test_H10) +'\n')
                
        if not_improve_epochs >= patience:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='FB15k-237', help='Dataset to use: {FB15k-237, YAGO3-10, WN18RR, umls, nations, kinship}, default: FB15k-237')
    parser.add_argument('--l2', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--model', type=str, default='conve', help='Choose from: {conve, distmult, complex}')
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--loader-threads', type=int, default=4, help='How many loader threads to use for the batch loaders. Default: 4')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the dataset. Needs to be executed only once. Default: 4')
    parser.add_argument('--resume', action='store_true', help='Resume a model.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    parser.add_argument('--spe_name', type=str, default=datetime.datetime.now().strftime("%m%d%H%M%S"), help='define a specific name for current model')
    parser.add_argument('--CFR_kernels', type=int, default=16, help='The number of kernels for Convolutional Feature Revision. Default: 16.')

    args = parser.parse_args()



    # parse console parameters and set global variables
    Config.backend = 'pytorch'
    Config.cuda = True
    Config.embedding_dim = args.embedding_dim
    #Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG


    model1_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    model1_path = 'saved_models/{0}_{1}_{2}.model1'.format(args.data, model1_name, args.spe_name)
    model2_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    model2_path = 'saved_models/{0}_{1}_{2}.model2'.format(args.data, model2_name, args.spe_name)

    torch.manual_seed(args.seed)
    main(args, model1_path, model2_path)
