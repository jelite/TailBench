#!/bin/bash

nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --sample=none \
    -o mistral_profile \
    bash ./run_model.sh