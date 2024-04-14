import numpy as np
from gym_torcs import TorcsEnv
from replay_buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork

# Assuming that ActorNetwork and CriticNetwork classes have been imported
# from actor import ActorNetwork
# from critic import CriticNetwork

class GaussianNoise:
    def __init__(self, action_dimension, sigma=0.2):
        self.action_dimension = action_dimension
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(0, self.sigma, self.action_dimension)

def play(train_indicator=1):
    buffer_size = 100000
    batch_size = 32
    gamma = 0.99  # discount factor
    tau = 0.001  # for soft update of target parameters
    lra = 0.0001  # learning rate for actor
    lrc = 0.001   # learning rate for critic

    action_dim = 1  # single continuous action
    state_dim = 21  # dimension of the state space in TORCS

    env = TorcsEnv(vision=False, throttle=False, gear_change=False)
    actor = ActorNetwork(state_size=state_dim, action_size=action_dim, hidden_units=[300, 600], tau=tau, lr=lra)
    critic = CriticNetwork(state_size=state_dim, action_size=action_dim, hidden_units=[300, 600], tau=tau, lr=lrc)
    buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    noise = GaussianNoise(action_dim)

    for episode in range(100):
        state = env.reset()
        total_reward = 0

        for step in range(1000):
            action = actor.network.forward_propagate(state.reshape(1, state_dim)) + noise()
            new_state, reward, done, _ = env.step(action.flatten())

            buffer.add((state, action, reward, new_state, done))
            total_reward += reward
            state = new_state

            if buffer.cur_size > batch_size:
                batch = buffer.get_batch(batch_size)
                states = np.array([e[0] for e in batch])
                actions = np.array([e[1] for e in batch])
                rewards = np.array([e[2] for e in batch])
                new_states = np.array([e[3] for e in batch])
                dones = np.array([e[4] for e in batch])

                target_actions = actor.target_network.forward_propagate(new_states)
                target_q_values = critic.target_network.forward_propagate(np.hstack([new_states, target_actions]))

                y = np.asarray([rewards[i] + gamma * target_q_values[i] * (1 - dones[i]) for i in range(batch_size)])

                critic.train(states, actions, y)
                action_gradients = np.array(critic.compute_action_gradients(states, actor.network.forward_propagate(states)))
                actor.train(states, action_gradients)
                actor.update_target_network()
                critic.update_target_network()

            if done:
                break

        print("Episode {}, Total Reward: {}".format(episode+1,total_reward))

if __name__ == "__main__":
    play(1)
