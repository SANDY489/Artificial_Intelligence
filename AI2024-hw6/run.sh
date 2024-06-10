#!/bin/bash

python main.py \
    --exp_name ORPO \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --train \
    --wandb_token 15d8c1222fd9699adc84f2baa9ac2413066c37ac \
    --num_epochs 1 \
