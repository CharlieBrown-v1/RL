from arguments import get_args
from algo import *
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from env import Make_Env
from gym_minigrid.wrappers import *

loops = 10
converge_threshold = 10
reward_threshold = 87


def single_run(n, start_planning):
    envs = Make_Env(env_mode=2)
    action_shape = envs.action_shape
    observation_shape = envs.state_shape

    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    width, height = np.array(envs.observation_shape[1:]) // envs.grid_size
    state_count = 2

    epsilon = 0.2
    alpha = 0.2
    gamma = 0.99

    steps_list = []
    times_list = []
    for _ in np.arange(loops):
        count = 0
        start = time.time()
        agent = myQAgent(width, height, state_count)
        Q = np.zeros((width * height * state_count, action_shape))
        model = DynaModel(width, height, state_count, action_shape, agent)
        for i in range(num_updates):
            obs = envs.reset()
            obs = obs.astype(int)
            for step in range(args.num_steps):
                # Sample actions with epsilon greedy policy

                if np.random.rand() < epsilon:
                    action = envs.action_sample()
                else:
                    action = agent.select_action(obs)

                # interact with the environment
                obs_next, reward, done, info = envs.step(action)
                obs_next = obs_next.astype(int)
                # add your Q-learning algorithm
                index = agent.obs2index(obs)
                index_next = agent.obs2index(obs_next)
                Q[index, action] += alpha * (reward + (1 - done) * gamma * np.max(Q[index_next]) - Q[index, action])
                agent.update(obs, Q[index].argmax())

                model.store_transition(obs, action, reward, obs_next)
                obs = obs_next

                if done:
                    obs = envs.reset()

            if i > start_planning:
                for _ in range(n):
                    s, idx = model.sample_state()
                    a = model.sample_action(s)
                    s_ = model.predict(s, a)
                    r = envs.R(s, a, s_)
                    done = envs.D(s, a, s_)
                    # add your Q-learning algorithm
                    index = agent.obs2index(s)
                    index_ = agent.obs2index(s_)
                    Q[index, a] += alpha * (
                                r + (1 - done) * gamma * np.max(Q[index_])
                                - Q[index, a])
                    if done:
                        break

            if (i + 1) % args.log_interval == 0:
                total_num_steps = (i + 1) * args.num_steps
                obs = envs.reset()
                obs = obs.astype(int)
                reward_episode_set = []
                reward_episode = 0.
                for step in range(args.test_steps):
                    action = agent.select_action(obs)
                    obs_next, reward, done, info = envs.step(action)
                    reward_episode += reward
                    obs = obs_next
                    if done:
                        reward_episode_set.append(reward_episode)
                        reward_episode = 0.
                        obs = envs.reset()

                end = time.time()
                print(
                    "TIME {} Updates {}, num timesteps {}, FPS {} \n avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
                        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                        i + 1, total_num_steps, int(total_num_steps / (end - start)),
                        np.mean(reward_episode_set),
                        np.min(reward_episode_set),
                        np.max(reward_episode_set)))

                min_reward = np.min(reward_episode_set)
                if min_reward < reward_threshold:
                    count = 0
                else:
                    count += 1
                if count >= converge_threshold:
                    steps_list.append(total_num_steps)
                    times_list.append(time.time() - start)
                    break

    return np.mean(steps_list), np.mean(times_list)


if __name__ == "__main__":
    steps_list = []
    time_list = []
    n_arr = np.arange(0, 200 + 1, 10)
    for n in n_arr:
        res = single_run(n, -1)
        steps_list.append(res[0])
        time_list.append(res[1])
