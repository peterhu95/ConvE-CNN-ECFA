#!/bin/bash
mkdir data
mkdir data/WN18RR
mkdir data/YAGO3-10
mkdir data/NELL-995
mkdir saved_models
tar -xvf WN18RR.tar.gz -C data/WN18RR
tar -xvf YAGO3-10.tar.gz -C data/YAGO3-10
tar -xvf NELL-995.tar.gz -C data/NELL-995
python wrangle_KG.py WN18RR
python wrangle_KG.py YAGO3-10
python wrangle_KG.py NELL-995
