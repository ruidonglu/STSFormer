#!/bin/bash
python experiments/train.py -c baselines/STSFormer/STSFormer_PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STSFormer/STSFormer_PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STSFormer/STSFormer_PEMS08.py --gpus '0'
