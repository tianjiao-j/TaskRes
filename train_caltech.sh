#!/bin/bash

CUDA_VISIBLE_DEVICES=1

bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep200 ./strong_base/Caltech101/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep200 ./strong_base/Caltech101/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep100 ./strong_base/Caltech101/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep100 ./strong_base/Caltech101/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep100 ./strong_base/Caltech101/rn50_1shots/model.pth.tar 1 0.5