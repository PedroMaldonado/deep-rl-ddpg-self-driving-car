import numpy as np
import time
from actor import ActorNetwork
from critic import CriticNetwork
from replay_buffer import ReplayBuffer


np.random.seed(1337)

# Define a fake TORCS environment for demonstration
class TorcsEnv:
    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

    def reset(self, relaunch=False):
        # Return a dummy observation
        return np.random.rand(21)  # Assume there are 21 state features

    def step(self, action):
        # Return dummy values for next_state, reward, done, info
        next_state = np.random.rand(21)
        reward = np.random.rand()
        done = np.random.choice([True, False], p=[0.1, 0.9])
        info = {}
        return next_state, reward, done, info

    def end(self):
        print("Environment closed.")

def play(train_indicator):
    buffer_size = 100000
    batch_size = 32
    gamma = 0.99
    tau = 0.001
    lra = 0.0001
    lrc = 0.001

    action_dim = 1
    state_dim = 21

    episodes_num = 10
    max_steps = 1000

    env = TorcsEnv()
    actor = ActorNetwork(state_dim, action_dim, [5, 5], tau, lra)
    critic = CriticNetwork(state_dim, action_dim, [5, 5], tau, lrc)
    buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    for episode in range(episodes_num):
        # print("Episode : {} Replay buffer {}".format(episode, len(buffer)))
        # state = env.reset(relaunch=(episode % 3 == 0))

        ob = env.reset(relaunch=True if i % 3 == 0 else False)

        state = np.hstack((ob.angle, ob.track, ob.trackPos))

        total_reward = 0

        for step in range(max_steps):
            action_predicted = actor.network.forward_propagate(state)[-1]  # Get action from actor
            action_noisy = action_predicted + np.random.normal(0, 0.1, size=action_dim)

            observation, reward, done, info = env.step(action_noisy[0])
            next_state = np.hstack((observation.angle, observation.track, observation.trackPos))
            buffer.add((state, action_noisy[0], reward, next_state, done))
            
            # next_state, reward, done, _ = env.step(action.flatten())
            # buffer.add((state, action, reward, next_state, done))

            if len(buffer) >= batch_size:
                batch = buffer.get_batch(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Prepare target values
                target_actions = [actor.target_network.forward_propagate(ns)[-1] for ns in next_states]
                target_values = [critic.target_network.forward_propagate(np.hstack((ns, ta)))[-1] for ns, ta in zip(next_states, target_actions)]
                y = rewards + gamma * np.where(dones, 0, np.squeeze(target_values))

                critic.train(np.array(states), np.array(actions), np.array(y))
                action_gradients = critic.get_action_gradients(np.array(states), np.array(actions))
                actor.train(np.array(states), action_gradients)

                actor.update_target_network()
                critic.update_target_network()

            state = next_state
            if done:
                break

        if train_indicator and episode % 3 == 0:
            print(f"Episode {episode+1}: Total Steps {step+1}, Training completed.")

    env.end()

if __name__ == "__main__":
    play(train_indicator=1)
