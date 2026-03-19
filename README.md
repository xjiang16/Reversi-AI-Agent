# Reversi AI Agent (Deep Q-Network)

## Overview
This project explores reinforcement learning by building an AI agent to play Reversi (Othello). The goal was to develop a model that can outperform a greedy baseline opponent by learning long-term strategies through self-play.

Our team implemented a Deep Q-Network (DQN) using PyTorch, applying core reinforcement learning techniques such as experience replay and epsilon-greedy exploration.

## Problem
In Reversi, a greedy strategy (maximizing immediate piece flips) often leads to poor long-term outcomes.

The objective of this project was to:
- move beyond short-term heuristics
- learn optimal strategies through interaction with the environment
- outperform a provided greedy baseline agent

## Approach
We modeled the problem as a reinforcement learning task:

- State: 8x8 board (flattened to 64 features)
- Action space: 64 possible board positions
- Reward: change in board advantage (piece count difference)
- Policy: Deep Q-Network (DQN)

Key techniques used:
- Deep Q-Learning
- Experience replay (replay buffer)
- Epsilon-greedy exploration
- Self-play training
- Neural network function approximation (PyTorch)

## Model Architecture
The DQN takes the board state as input and outputs Q-values for all possible moves.

- Fully connected neural network
- Input: 64 (flattened board)
- Hidden layers: ReLU activations
- Output: 64 Q-values (one per action)

## Results
The trained agent consistently outperforms the greedy baseline, demonstrating improved long-term decision-making and stronger positional strategy compared to short-term heuristics.

## Repository Structure
Reversi-AI-Agent/
│
├── provided/                     # Professor-provided code (environment + baseline)
│   ├── reversi_pieces_png/       # Game assets (board + pieces)
│   │   ├── background.jpeg
│   │   ├── black_piece.png
│   │   └── white_piece.png
│   ├── greedy_player.py
│   ├── reversi.py
│   └── reversi_server.py
│
├── src/                          # Team-developed RL agent
│   ├── dqn_training.py
│   └── dqn_player.py
│
├── README.md
└── .gitignore

## Key Files
Provided (not authored by me):
- greedy_player.py: baseline opponent using greedy strategy
- reversi.py: game logic
- reversi_server.py: game server and gameplay interface
- reversi_pieces_png/: visual assets used by the game UI

Team Implementation:
- dqn_training.py: DQN training loop, replay memory, and optimization logic
- dqn_player.py: Reversi environment and DQN-based agent implementation

## How to Run
1. Install dependencies:
pip install torch numpy

2. Train the agent:
python src/dqn_training.py

3. Run alternative implementation:
python src/dqn_player.py

## What I Learned
This project helped me develop practical experience with:
- reinforcement learning fundamentals
- Deep Q-Network implementation in PyTorch
- balancing exploration vs exploitation
- designing reward functions
- working with game environments
- debugging training instability in RL

## Limitations
This is an early-stage reinforcement learning project and has several limitations:
- limited hyperparameter tuning
- relatively simple neural network architecture
- no advanced techniques such as Double DQN or prioritized replay
- limited evaluation metrics (no systematic win-rate tracking)

## Future Improvements
- implement Double DQN or dueling architectures
- improve reward shaping
- track win rate across games
- tune hyperparameters more systematically
- evaluate against stronger opponents

## Notes
This was a team project, and part of the codebase (environment, assets, and baseline agent) was provided by the instructor. The reinforcement learning agent and training logic were developed by our team.
