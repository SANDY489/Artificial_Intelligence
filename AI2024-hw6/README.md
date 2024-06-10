# Homework 6

## Create Conda Env

conda create -y -n ai_hw6 python=3.10

conda activate ai_hw6

## Install Necessary Packages

pip install torch 
pip install --no-deps trl peft accelerate bitsandbytes
pip install tqdm packaging wandb

conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps trl peft accelerate bitsandbytes

## Implementation

bash run.sh

bash inference.sh