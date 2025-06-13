# NEAT Flappy Bird ![bird1](https://github.com/user-attachments/assets/a801826e-16dc-4370-891d-d7eeae0ee346)



An AI that learns to play Flappy Bird using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm, implemented with Pygame.


## Features ✨

- 🧠 AI learns to play Flappy Bird from scratch using neural networks
- 🔄 Evolutionary algorithm that improves generation after generation
- 🎮 Pygame-based visualization of the learning process
- 🏆 Save and replay the best performing bird
- ⚙️ Configurable parameters for evolution and gameplay

## Requirements 📋

- Python 3.6+
- Pygame
- NEAT-Python

## Installation ⚙️

1. Clone the repository:
```bash
git clone https://github.com/Ehshanulla/Neat-Flappy-Bird.git
cd Neat-Flappy-Bird
```

# Install dependencies:
  ```bash
  pip install -r requirements.txt
```

# Usage 🚀
Training the AI
```bash
# Default training (50 generations)
python flappy_bird.py

# With custom target score
python flappy_bird.py --target_score 50

# With limited generations
python flappy_bird.py --generations 30
```
Watching the best bird play
After training completes, the best bird will automatically play. You can also run just the best bird:

```bash
# Run the saved winner
python play_winner.py

# Run with custom score limit
python play_winner.py --score_limit 75
```
# Configuration ⚙️
The NEAT algorithm can be configured by editing config-feedforward.txt:

- Population size

- Mutation rates

- Activation functions

- And other evolutionary parameters

# How It Works 🧠
1.Initialization: Creates a population of birds with random neural networks

2.Evaluation: Each bird plays the game, with fitness based on distance traveled

3.Selection: Best-performing birds are selected for reproduction

4.Reproduction: Offspring are created with mutations and crossover

5.Repeat: Process continues until stopping condition is met

# Results 📊
After training, you can:

- Visualize the best neural network

- Plot fitness over generations

- Save the best genome for later use

# Customization 🎨
You can modify:

- Game speed (clock.tick() value)

- Pipe gap size

- Bird physics (jump velocity, gravity)

- NEAT parameters in the config file

# Contributing 🤝
Contributions are welcome! Please open an issue or pull request for any:

- Bug fixes

- Performance improvements

- New features


