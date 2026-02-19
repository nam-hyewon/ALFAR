#!/bin/bash

source ~/.bashrc
conda activate alfar

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/attention_InstructBLIP.py \
    --model-path Salesforce/instructblip-vicuna-7b \
    --image-folder data/infoseek/images/infoseek \
    --num-samples 100
