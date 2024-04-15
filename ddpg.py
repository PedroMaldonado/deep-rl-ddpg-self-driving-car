import argparse
import time
import numpy as np

from actor_net import *
from critic_net import *
from ReplayBuffer import ReplayBuffer

from gym_torcs import TorcsEnv
from util.noise import OrnsteinUhlenbeckActionNoise

#semilla para random
np.random.seed(1337)

def play(train_indicator):
    buffer_size = 100000
    batch_size = 32
    MINIBATCH_SIZE = 64
    gamma = 0.99    # discount factor
    tau = 0.001     # Target Network HyperParameter
    lra = 0.0001    # Learning rate for Actor
    lrc = 0.001     # Learning rate for Critic
    ou_sigma = 0.3

    action_dim = 1  # Steering angle
    state_dim = 21  # num of sensors input

    episodes_num = 5
    max_steps = 100
    step = 0
    
    train_stat_file = "data/train_stat.txt"
    actor_weights_file = "data/actor.h5"
    critic_weights_file = "data/critic.h5"

    HIDDEN1_UNITS = 300
    HIDDEN2_UNITS = 600

    TAU = 0.001
    
    # Size of replay buffer
    ACTION_BOUND=2
    
    # Create actor and critic nets
    actor = ActorNet(state_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, action_dim)
    critic = CriticNet(state_dim, action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS,HIDDEN2_UNITS, action_dim)
    
    # Initialize replay buffer
    buff = ReplayBuffer(buffer_size)      
        
    # noise function for exploration
    # An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction
    ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=ou_sigma * np.ones(action_dim))
    
    # Torcs environment - throttle and gear change controlled by client
    env = TorcsEnv(vision=False, throttle=False, gear_change=False)

    reward_result = []
    
    for i in range(episodes_num):

        #print("Episode : %s Replay buffer %s" % (i, len(buff)))

        if i % 3 == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        # 21 len state dimensions - https://arxiv.org/abs/1304.1672
        state = np.hstack((ob.angle, ob.track, ob.trackPos))

        total_reward = 0.
        for j in range(max_steps):
            loss = 0

            action_predicted = actor.predict(state.reshape(1, state.shape[0]), ACTION_BOUND, target=False)+1./(1.+i+j)
            
            observation, reward, done, info = env.step(action_predicted[0])

            state1 = np.hstack((observation.angle, observation.track, observation.trackPos))

            buff.add(ob, action_predicted[0], reward, state1, done)  # add replay buffer

            #if buff.count() > MINIBATCH_SIZE:
                
            # batch update
            batch = buff.getBatch(batch_size)
            states_t = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            states_t_1 = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            #y_t = np.asarray([e[1] for e in batch])
            # Setup y_is for updating critic
            y_t=np.zeros((len(batch), action_dim))
            a_tgt=actor.predict(states_t_1, ACTION_BOUND, target=True)
            target_q_values = critic.predict(states_t_1, a_tgt,target=True)#critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + gamma * target_q_values[k]

            if train_indicator:
                # Update critic by minimizing the loss
                loss += critic.train(states_t, actions, y_t)
                
                # Update actor using sampled policy gradient
                a_for_grad = actor.predict(states_t, ACTION_BOUND, target=False)
                
                grads = critic.evaluate_action_gradient(states_t, a_for_grad)
                actor.train(states_t, grads, ACTION_BOUND)
                actor.train_target(TAU)
                critic.train_target(TAU)

            total_reward += reward
            state = state1

            print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action_predicted[0][0], reward))

            step += 1
            if done:
                break
        
        tm = time.strftime("%Y-%m-%d %H:%M:%S")
        episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, step, total_reward, tm)
        print(episode_stat)

        with open(train_stat_file, "a") as outfile:
            outfile.write(episode_stat+"\n")

    env.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=0)
    args = parser.parse_args()
    play(args.train)
