# Exponential and Linear Decay in DQN

This project demonstrates the implementation of **Exponential** and **Linear Decay** techniques in the **Deep Q-Network (DQN)** algorithm, which is used for reinforcement learning. These decay strategies are employed to reduce the exploration rate (epsilon) over time, making the agent gradually shift from exploration to exploitation as it learns.

## Features
- **Exponential Decay**: The epsilon value decreases exponentially over time, encouraging the agent to explore more initially and then exploit learned strategies as it progresses.
- **Linear Decay**: The epsilon value decreases linearly over time, offering a steady reduction in exploration, which can lead to smoother convergence.
- **DQN Implementation**: This project uses a Deep Q-Network (DQN) to solve reinforcement learning problems, such as classic control tasks like CartPole or MountainCar.

## Prerequisites
Ensure you have the following installed before running the code:
- Python 3.x
- PyTorch (for deep learning model)
- NumPy
- Gym (for reinforcement learning environments)
- Matplotlib (for visualizing the training progress)

You can install the required libraries using `pip`:

```bash
pip install numpy gym torch matplotlib
