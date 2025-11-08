#!/bin/bash

nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --sample=none \
    -o ./profile/test \
    python model_launch.py --batch 1 --model_name mistralai/Mistral-7B-Instruct-v0.3
