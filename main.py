import numpy as np
from Satellite_Envirnoment import LEO_RIS_NOMA_Env
from TD3 import TD3,ReplayBuffer
from Actor_Critic_Networks import Actor,Critic

def train_td3(env, agent, replay_buffer, episodes=100, batch_size=100):
    state = env.reset() # this must change every episode, but i putted it here for test
    for episode in range(episodes):
        # state = env.reset()
        episode_reward = 0

        for t in range(1000):
            action = agent.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)

            agent.train(replay_buffer, batch_size)

            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"Episode {episode+1}, Reward: {episode_reward}, power: {action[0]} + {action[1]}")



# Example usage:
# User and RIS positions (on Earth)
users_pos = np.array([[0, 0, 0], [0.008, 0.01, 0]])  # 2 users on Earth's surface
RIS_pos = np.array([1, 1, 50])                     # RIS on Earth's surface
# Assuming your environment is initialized as env

env = LEO_RIS_NOMA_Env()
env._generate_user_and_ris_positions(users_pos,RIS_pos)
env._Satellite_move(0)
env.user1_power = 0.3
env.user2_power =0.7

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

agent = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()

train_td3(env, agent, replay_buffer)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
