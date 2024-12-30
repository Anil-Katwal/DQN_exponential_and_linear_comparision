#GOAL: The agent must reach position 9 with as few moves as possible to maximize reward
#0: Starting position
#9: Goal position
#Actions: The agent moves left or right along the line based on the action it selects

#Start                                Goal
#  |                                   |
#  0---1---2---3---4---5---6---7---8---9

'''
Grid: Imagine a line with 10 cells (0-9). The agen starts on the far left.
Aims to reach the gaol on the far right

Actions: Left -> Action0, which decreases the position by 1, but it can't go left past 0.
         Right -> Action1, which increases the position by 1, but it can't go past 9.

Rewards: Each step results in a reward of -1, penalizing the agent to encourage
fewer steps (for a faster solution)
Reaching the goal (position 9) gives a rwards of +10.

Stopping criterion:
The episode ends when:
a. The agent reaches the goal
b. The agent hits the max number of steps allowed. 50 in this example.
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os

# Create a simple custom environment
class SimpleEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.state = 0  
        self.goal = size - 1  

    def reset(self):
        self.state = 0  
        return self.state

    def step(self, action):
        reward = -1  
        done = False

        if action == 1:  # Move right
            self.state = min(self.state + 1, self.size - 1)
        elif action == 0:  # Move left
            self.state = max(self.state - 1, 0)

        if self.state == self.goal: 
            reward = 20
            done = True

        return self.state, reward, done

# Build the DQN model
def build_model(input_shape, action_space):
    model = models.Sequential([
        layers.Dense(128, input_shape=(input_shape,), activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(action_space, activation="linear")
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss="mse")
    return model

# Linear Decay Function
def linear_decay(epsilon_start, epsilon_min, episode, total_episodes):
    decay_rate = (epsilon_start - epsilon_min) / total_episodes
    return max(epsilon_start - decay_rate * episode, epsilon_min)

# Exponential Decay Function
def exponential_decay(epsilon_start, epsilon_min, episode, decay_rate=0.995):
    return max(epsilon_start * (decay_rate ** episode), epsilon_min)

# Initialize environment and DQN
env = SimpleEnvironment(size=10)
state_size = 1  
action_size = 2  # Two possible actions: Left or Right
model = build_model(state_size, action_size)

# Hyperparameters
episodes = 500
gamma = 0.99  # Discount factor
epsilon_start = 1.0
epsilon_min = 0.01
decay_type = "exponential"
steps = 50  

# Logging with TensorBoard
log_dir = "./logs/dqn"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training loop
rewards = []
epsilon_values = []
actions_count = np.zeros((episodes, action_size))

for episode in range(episodes):
    # Update epsilon
    if decay_type == "linear":
        epsilon = linear_decay(epsilon_start, epsilon_min, episode, episodes)
    elif decay_type == "exponential":
        epsilon = exponential_decay(epsilon_start, epsilon_min, episode)
    epsilon_values.append(epsilon)

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(steps):
        # Choose action 
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_size)  # Explore
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])  # Exploit

        actions_count[episode, action] += 1

        # Take action and observe reward
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        # Update Q-values (Bellman equation)
        target = reward + gamma * np.amax(model.predict(next_state, verbose=0)[0]) if not done else reward
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target

        # Train the model
        model.fit(state, target_f, epochs=1, verbose=0, callbacks=[tensorboard_callback])

        state = next_state
        if done:
            break

    rewards.append(total_reward)
    print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward}")
# Plot results
linear_epsilons = [linear_decay(epsilon_start, epsilon_min, i, episodes) for i in range(episodes)]
exponential_epsilons = [exponential_decay(epsilon_start, epsilon_min, i) for i in range(episodes)]

# Plot epsilon decay
plt.figure(figsize=(10, 6))
plt.plot(linear_epsilons, label="Linear Decay", color="blue")
plt.plot(exponential_epsilons, label="Exponential Decay", color="red")
plt.title("Epsilon Decay Strategies")
plt.xlabel("Episodes")
plt.ylabel("Epsilon")
plt.legend()
plt.grid(True)
plt.savefig("epsilon_decay.png")
plt.show()

# Plot actions per episode
plt.figure(figsize=(10, 6))
plt.plot(actions_count[:, 0], label="Action 0 (Left)", color="purple")
plt.plot(actions_count[:, 1], label="Action 1 (Right)", color="orange")
plt.title("Actions Per Episode")
plt.xlabel("Episodes")
plt.ylabel("Count of Actions")
plt.legend()
plt.grid(True)
plt.savefig("actions_per_episode.png")
plt.show()

# Plot rewards over episodes
plt.figure(figsize=(10, 6))
plt.plot(rewards, label="Rewards per Episode", color="green")
plt.title("Rewards Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.savefig("rewards_over_episodes.png")
plt.show()

# TensorBoard visualization command
# Visualizing results with TensorBoard
# To view, use the following command in your terminal:
# tensorboard --logdir=./logs/dqn
