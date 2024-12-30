import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime

# Custom environment
class SimpleEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.state = 0
        self.goal = size - 1

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        reward = -0.5  # Reduced step penalty
        done = False

        if action == 1:  # Move right
            self.state = min(self.state + 1, self.size - 1)
        elif action == 0:  # Move left
            self.state = max(self.state - 1, 0)

        if self.state == self.goal:  # Goal reached
            reward = 20  # Increased goal reward
            done = True

        return self.state, reward, done


# Build DQN model
def build_model(input_shape, action_space):
    model = models.Sequential([
        layers.Dense(128, input_shape=(input_shape,), activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(action_space, activation="linear")
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003), loss="mse")
    return model


# Epsilon Decay Functions
def linear_decay(epsilon_start, epsilon_min, episode, total_episodes):
    decay_rate = (epsilon_start - epsilon_min) / total_episodes
    return max(epsilon_start - decay_rate * episode, epsilon_min)


def exponential_decay(epsilon_start, epsilon_min, episode, decay_rate=0.997):
    return max(epsilon_start * (decay_rate ** episode), epsilon_min)


# Initialize the environment and model
env = SimpleEnvironment(size=10)
state_size = 1
action_size = 2
model = build_model(state_size, action_size)

# TensorBoard setup
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="batch")

# Hyperparameters
episodes = 500  # Increased episodes
gamma = 0.98  # Discount factor
epsilon_start = 1.0
epsilon_min = 0.05
decay_type = "exponential"
max_steps = 100  # Increased steps per episode

# Training loop
rewards = []
epsilon_values = []
for episode in range(episodes):
    if decay_type == "linear":
        epsilon = linear_decay(epsilon_start, epsilon_min, episode, episodes)
    elif decay_type == "exponential":
        epsilon = exponential_decay(epsilon_start, epsilon_min, episode)
    epsilon_values.append(epsilon)

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(max_steps):
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_size)  # Explore
        else:
            q_values = model.predict(state, verbose=0)  # Exploit
            action = np.argmax(q_values[0])

        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        target = reward + gamma * np.amax(model.predict(next_state, verbose=0)[0]) if not done else reward
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target

        model.fit(state, target_f, epochs=1, verbose=0, callbacks=[tensorboard_callback])

        state = next_state

        if done:
            print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward}, Steps: {step + 1}")
            rewards.append(total_reward)
            break

# Plot epsilon decay
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, label="Epsilon Decay")
plt.title("Epsilon Decay Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Epsilon")
plt.legend()
plt.grid(True)
plt.savefig('Epsilon_Decay_Updated.png')
plt.show()

# Plot rewards over episodes
plt.figure(figsize=(10, 6))
plt.plot(rewards, color='green', label="Rewards per Episode")
plt.title("Rewards Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.savefig('Rewards_Updated.png')
plt.show()

# TensorBoard instructions
print(f"To view training progress, run: tensorboard --logdir={log_dir}")
