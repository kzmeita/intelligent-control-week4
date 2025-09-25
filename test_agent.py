import gym 
import numpy as np
from dqn_agent import DQNAgent

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
agent.model.load_weights("dqn_cartpole.h5")  
agent.epsilon = 0.01 
for e in range(5):
    state, _ = env.reset() 
    state = np.reshape(state, [1, state_size])

    for time in range(500):
        env.render()
        action = agent.act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = np.reshape(next_state, [1, state_size])
        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            break

env.close()
