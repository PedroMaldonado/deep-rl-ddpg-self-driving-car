import argparse
import numpy as np
import time
from gym_torcs import TorcsEnv
from actor import ActorNetwork
from critic import CriticNetwork
from replay_buffer import ReplayBuffer

np.random.seed(1337)

def play(train_indicator):
    buffer_size = 100000
    batch_size = 32
    gamma = 0.99  # discount factor
    tau = 0.001  # Target Network HyperParameter
    lra = 0.0001  # Learning rate for Actor
    lrc = 0.001  # Learning rate for Critic

    action_dim = 1  # Steering angle
    state_dim = 21  # num of sensors input

    episodes_num = 10
    max_steps = 1000

    actor = ActorNetwork(state_size=state_dim, action_size=action_dim, hidden_units=[5, 5], tau=tau, lr=lra)
    critic = CriticNetwork(state_size=state_dim, action_size=action_dim, hidden_units=[5, 5], tau=tau, lr=lrc)
    buffer = ReplayBuffer(max_size=buffer_size, input_shape=state_dim, n_actions=action_dim)

    env = TorcsEnv(vision=False, throttle=False, gear_change=False)

    for i in range(episodes_num):
        print("Episode : {} Replay buffer {}".format(i, len(buffer)))
        ob = env.reset(relaunch=True if i % 3 == 0 else False)

        state = np.hstack((ob.angle, ob.track, ob.trackPos))
        total_reward = 0

        for j in range(max_steps):
            action_predicted = actor.network.forward_propagate(state.reshape(-1, 1))  # Forward pass
            action_noisy = action_predicted + np.random.normal(0, 0.1, size=action_dim)  # Add some noise for exploration

            observation, reward, done, info = env.step(action_noisy[0])
            new_state = np.hstack((observation.angle, observation.track, observation.trackPos))
            buffer.add((state, action_noisy[0], reward, new_state, done))

            if len(buffer) > batch_size:
                batch = buffer.get_batch(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = np.array(states)
                actions = np.array(actions).reshape(-1, 1)
                rewards = np.array(rewards)
                # next_states = np.array(next_states)
                next_states = np.array(next_states)
                next_states = next_states.reshape(-1, state_dim)
                dones = np.array(dones)

                print("Shape of next_states: ", next_states.shape)
                next_actions = actor.target_network.forward_propagate(next_states)
                print("Shape of next_actions: ", next_actions.shape)

                # Ensure next_actions is properly shaped
                if next_actions.shape[0] != next_states.shape[0]:
                    next_actions = next_actions.reshape(next_states.shape[0], -1)  # Reshape next_actions to match next_states in the first dimension

                # Now concatenate along axis 1 (columns)
                combined_state_action = np.hstack((next_states, next_actions))
                print("Shape of combined state-action: ", combined_state_action.shape)

                target_q_values = critic.target_network.forward_propagate(combined_state_action)

                y_t = rewards + gamma * np.multiply((1 - dones), target_q_values)

                critic.train(states, actions, y_t)
                action_for_grad = actor.network.forward_propagate(states)
                grads = critic.compute_action_gradients(states, action_for_grad)
                actor.train(states, grads)

                actor.update_target_network()
                critic.update_target_network()

            total_reward += reward
            state = new_state

            if done:
                break

        print("Episode: {}, Total Reward: {}".format(i, total_reward))

    env.end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=0)
    args = parser.parse_args()
    play(args.train)
