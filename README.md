# ConvE-CFR
Source code for model ConvE-CFR
# Running the model:
The data needed to preprocess once. The --preprocess parameter can be omitted after the first time.
WN18RR:
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --CFR_kernels 32 --lr 0.001 

YAGO3-10:
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data YAGO3-10 --lr 0.001

NELL-995:
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data NELL-995 --lr 0.001 --test-batch-size 64 --batch-size 64 --preprocess

# Acknowledgements
The code is inspired by [ConvE](https://github.com/TimDettmers/ConvE).
