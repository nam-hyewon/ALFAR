#!/bin/bash

source ~/.bashrc
conda activate alfar

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/alfar_mc_qwen.py \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --image-folder data/infoseek/images/infoseek \
    --dataset infoseek \
    --answers-file ./results/qwen_alfar_run5.jsonl \
    --start-index 2178
