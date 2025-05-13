# Dissertation - Optimising the AlphaZero Reinforcement Learning Framework for Impartial Games Using Weak Neural Networks: A Case Study in NIM With Multi-Frame Inputs 


## Setup

```bash
pip install -r requirements.txt
```

## Training

Train a model with or without history:
```bash
python main.py --include_history 1  # With history
python main.py --include_history 0  # Without history
```

Key options:
- `--alpha`: Exploration noise (default: 1.0)
- `--iterations`: Training iterations (default: 300)
- `--simulations`: MCTS sims per move (default: 100)
- `--self_play_games`: Games per iteration (default: 200)
- `--random_starts`: Use varied starting positions

## Evaluation

Compare models:
```bash
python tournament.py --games 200  # Run 200 games per board
```

Plot training progress:
```bash
python plot_training_curves.py --alpha 1.0
```

## Game Rules

Nim is played with piles of counters. Players take turns removing any number of counters from a single pile. The player who takes the last counter wins.

Default board: [1,3,5,7,9]

## Project Structure

- `main.py`: Training script
- `game.py`: Nim game logic
- `model.py`: Neural network architecture
- `mcts.py`: Monte Carlo Tree Search
- `trainer.py`: Training loop and self-play
- `tournament.py`: Model comparison
- `analysis.py`: Performance analysis