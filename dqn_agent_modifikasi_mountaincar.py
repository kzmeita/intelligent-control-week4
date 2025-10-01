import gymnasium as gym
import numpy as np
import random
import pygame
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Pygame setup
def init_pygame():
    pygame.init()
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MountainCar DQN")
    font = pygame.font.SysFont('comic sans ms', 24)
    return screen, font

# Render environment using Pygame
def render_pygame(screen, font, frame, score, episode, epsilon):
    screen.fill((255, 255, 255))

    if frame is not None:
        frame[(frame[:, :, 0] == 255) & (frame[:, :, 1] == 255) & (frame[:, :, 2] == 0)] = [255, 0, 0]

        frame = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        frame = pygame.transform.scale(frame, (500, 400))
        screen.blit(frame, (70, 20))

    score_text = font.render(f"Score: {score:.2f}", True, (0, 0, 0))
    episode_text = font.render(f"Episode: {episode}", True, (0, 0, 0))
    epsilon_text = font.render(f"Epsilon: {epsilon:.2f}", True, (0, 0, 0))
    exit_text = font.render("Training running...", True, (0, 255, 0))
    
    screen.blit(score_text, (20, 430))
    screen.blit(episode_text, (220, 430))
    screen.blit(epsilon_text, (420, 430))
    screen.blit(exit_text, (240, 460))

    pygame.display.flip()

# Setup grafik
plt.ion()
fig, ax = plt.subplots()
episodes_list = []
steps_list = []
scores_list = []
epsilons_list = []

def update_plot(episode, steps, score, epsilon):
    episodes_list.append(episode)
    steps_list.append(steps)
    scores_list.append(score)
    epsilons_list.append(epsilon)
    
    ax.clear()
    ax.plot(episodes_list, steps_list, label='Steps', color='blue')
    ax.plot(episodes_list, scores_list, label='Score', color='green')
    ax.plot(episodes_list, epsilons_list, label='Epsilon', color='red')

    ax.set_title('Training Progress')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.legend()
    plt.pause(0.01)

# Main training loop
if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 100
    batch_size = 32

    screen, font = init_pygame()

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        steps = 0

        for time in range(2000):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            reward = -1 if not done else 100
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            steps += 1

            frame = env.render()
            render_pygame(screen, font, frame, score, e + 1, agent.epsilon)

            # Update grafik setiap step
            update_plot(e + 1, steps, score, agent.epsilon)

            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {score:.2f}, Epsilon: {agent.epsilon:.2f}, Steps: {steps}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    print("\n Training selesai! Menutup program dalam 3 detik...")
    pygame.time.wait(500)

    env.close()
    pygame.quit()
    plt.ioff()
    plt.show()