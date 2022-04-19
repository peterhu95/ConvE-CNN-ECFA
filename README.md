
# ConvE-CNN-ECFA
Source code for model ConvE-CNN-ECFA
## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch 1.0 ](https://github.com/pytorch/pytorch) using [official website](https://pytorch.org/) or [Anaconda](https://www.continuum.io/downloads). 

2. Install the requirements: `pip install -r requirements.txt`

3. Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step `python -m spacy download en`.
# Running the model:
The data needed to preprocess once. The --preprocess parameter can be omitted after the first time.

WN18RR:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --preprocess --CFR_kernels 32 --lr 0.001
```

YAGO3-10:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data YAGO3-10 --preprocess --lr 0.001
```

NELL-995:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data NELL-995 --preprocess --lr 0.001 --test-batch-size 64 --batch-size 64
```



