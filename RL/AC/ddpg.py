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

    episodes_num = 2
    max_steps = 10

    actor = ActorNetwork(state_size=state_dim, action_size=action_dim, hidden_units=[300, 600], tau=tau, lr=lra)
    critic = CriticNetwork(state_size=state_dim, action_size=action_dim, hidden_units=[300, 600], tau=tau, lr=lrc)
    buffer = ReplayBuffer(buffer_size)

    env = TorcsEnv(vision=False, throttle=False, gear_change=False)

    for i in range(episodes_num):
        print("Episode : %s Replay buffer %s" % (i, len(buffer)))
        ob = env.reset(relaunch=True if i % 3 == 0 else False)

        state = np.hstack((ob.angle, ob.track, ob.trackPos))
        total_reward = 0

        for j in range(max_steps):
            action_predicted = actor.network.forward_propagate(state.reshape(1, -1))  # Forward pass
            action_noisy = action_predicted + np.random.normal(0, 0.1, size=action_dim)  # Add some noise for exploration

            observation, reward, done, info = env.step(action_noisy[0])
            new_state = np.hstack((observation.angle, observation.track, observation.trackPos))
            buffer.add((state, action_noisy[0], reward, new_state, done))

            if len(buffer) > batch_size:
                batch = buffer.get_batch(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert data to NumPy arrays for batch processing
                states = np.array(states)
                actions = np.array(actions).reshape(-1, 1)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)

                # Compute target Q values
                next_actions = actor.target_network.forward_propagate(next_states)
                target_q_values = critic.target_network.forward_propagate(np.hstack((next_states, next_actions)))

                y_t = rewards + gamma * np.multiply((1 - dones), target_q_values)

                # Update critic by minimizing the loss
                critic.train(states, actions, y_t)

                # Update actor policy using the sampled gradient
                action_for_grad = actor.network.forward_propagate(states)
                grads = critic.compute_action_gradients(states, action_for_grad)
                actor.train(states, grads)

                # Soft update target networks
                actor.update_target_network()
                critic.update_target_network()

            total_reward += reward
            state = new_state

            if done:
                break

        print(f"Episode: {i}, Total Reward: {total_reward}")

    env.end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=0)
    args = parser.parse_args()
    play(args.train)