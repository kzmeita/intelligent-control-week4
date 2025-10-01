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
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
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
    pygame.display.set_caption("CartPole DQN")
    font = pygame.font.SysFont('Arial', 24)
    return screen, font

# Render environment using Pygame
def render_pygame(screen, font, frame, score, episode, epsilon):
    screen.fill((255, 255, 255))

    if frame is not None:
        frame = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        frame = pygame.transform.scale(frame, (500, 400))
        screen.blit(frame, (70, 20))

    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    episode_text = font.render(f"Episode: {episode}", True, (0, 0, 0))
    epsilon_text = font.render(f"Epsilon: {epsilon:.2f}", True, (0, 0, 0))
    exit_text = font.render("Press 'Q' to Quit", True, (255, 0, 0))
    
    screen.blit(score_text, (20, 430))
    screen.blit(episode_text, (220, 430))
    screen.blit(epsilon_text, (420, 430))
    screen.blit(exit_text, (240, 460))

    pygame.display.flip()

# Main training loop
if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    screen, font = init_pygame()

    # Untuk menyimpan data episode dan score untuk plotting
    episode_list = []
    score_list = []
    epsilon_list = []

    # Konfigurasi matplotlib untuk real-time plot
    plt.ion()
    fig, ax = plt.subplots()
    line_score, = ax.plot([], [], label='Score', color='blue')
    line_epsilon, = ax.plot([], [], label='Epsilon', color='red')
    ax.set_xlim(0, episodes)
    ax.set_ylim(0, 500)
    ax.set_title('Training Performance')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.legend()
    plt.grid()

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0

        for time in range(500):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score = time

            frame = env.render()
            render_pygame(screen, font, frame, score, e + 1, agent.epsilon)

            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
                episode_list.append(e + 1)
                score_list.append(score)
                epsilon_list.append(agent.epsilon)

                # Perbarui data plot
                line_score.set_xdata(episode_list)
                line_score.set_ydata(score_list)
                line_epsilon.set_xdata(episode_list)
                line_epsilon.set_ydata(epsilon_list)

                # Perbarui skala jika perlu
                ax.relim()
                ax.autoscale_view()

                # Perbarui grafik
                plt.pause(0.01)
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    plt.ioff()
    plt.show()

    # Tunggu setelah episode terakhir selesai
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    waiting = False

    env.close()
    pygame.quit()