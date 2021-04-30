# IFT 6756 - Final Project

Here, I explore AlphaZero, the work of [David Silver et al. [2018]](https://science.sciencemag.org/content/362/6419/1140),
which is a general reinforcement learning algorithm. Specifically, I will be exploring the MCTS by tackling a 9x9 version Go.

## Setup:

Use a python3 environment to:
- pip install -r requirements.txt
- Install https://github.com/aigagror/GymGo by following their instructions
- Fetch `elopy.py` from https://github.com/HankSheehan/EloPy

## Usage:

- `training.py` is used to train AlphaZero. It alternates between generating self-play games and training on the data produced. Trained models are placed in a `checkpoints/` folder. Choose whether to optimize for Z values or Q values by setting the target_mode variable within to either TargetMode.Z_VALUES or TargetMode.Q_VALUES.
- `evaluation.py` is used to evaluate AlphaZero against a random agent. It plays competition games and reports on the ELO ratings of both players. Load a model here that was created by the training script.

## Disclaimer:

I used DeepMind's pseudocode as a base for implementing AlphaZero, available in the supplementary **Data S1** available [here](https://science.sciencemag.org/content/362/6419/1140/tab-figures-data).
