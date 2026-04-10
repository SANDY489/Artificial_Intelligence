# Homework 5

## Introduction

In this project, we implement a **Deep Q-Network (DQN)** agent to play **OpenAI Gymnasium ALE/MsPacman-v5**. The main objective is to train an agent that can observe image-based game states, choose actions, and gradually learn a better policy through reinforcement learning.

To achieve this, we build a **PacmanActionCNN** for extracting visual features and predicting action values, and we implement the main DQN components in `rl_algorithm.py`, including replay memory, epsilon-greedy exploration, target-network updates, and Q-value learning. In addition, we complete the `train`, `validation`, and `evaluate` functions in `pacman.py` so the whole training and testing pipeline can run correctly.

The overall workflow of this project is: set up the required environment, train the DQN agent on MsPacman, track the reward and loss during training, and evaluate the final model by running it in the game environment. This README describes the environment setup and the steps needed to reproduce both training and evaluation. 

## Install Necessary Packages
conda create -n hw5 python=3.11 -y
conda activate hw5
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python3 pacman.py
