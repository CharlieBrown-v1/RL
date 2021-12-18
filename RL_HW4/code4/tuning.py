from arguments import get_args
from algo import *
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from env import Make_Env
from gym_minigrid.wrappers import *
import optuna


loops = 1
converge_threshold = 10
reward_threshold = 87
INF = 0x3f3f3f3f


def single_run(n, start_planning, h, m):
    envs = Make_Env(env_mode=2)
    action_shape = envs.action_shape

    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    width, height = np.array(envs.observation_shape[1:]) // envs.grid_size
    state_count = 2

    epsilon = 0.2
    alpha = 0.2
    gamma = 0.99
    
    steps_list = []
    for _ in np.arange(loops):
        count = 0
        agent = myQAgent(width, height, state_count)
        Q = np.zeros((width * height * state_count, action_shape))
        dynamics_model = NetworkModel(8, 8, policy=agent)
        start = time.time()
        for i in range(num_updates):
            obs = envs.reset()
            obs = obs.astype(int)
            for step in range(args.num_steps):
                if np.random.rand() < epsilon:
                    action = envs.action_sample()
                else:
                    action = agent.select_action(obs)

                obs_next, reward, done, info = envs.step(action)
                obs_next = obs_next.astype(int)
                # add your Q-learning algorithm
                index = agent.obs2index(obs)
                index_next = agent.obs2index(obs_next)
                Q[index, action] += alpha * (reward + (1 - done) * gamma * np.max(Q[index_next]) - Q[index, action])
                agent.update(obs, Q[index].argmax())
    
                dynamics_model.store_transition(obs, action, reward, obs_next)
                obs = obs_next
    
                if done:
                    obs = envs.reset()

            for _ in range(m):
                dynamics_model.train_transition(32)

            if i > start_planning:
                for _ in range(n):
                    s, idx = dynamics_model.sample_state()
                    for _ in range(h):
                        if np.random.rand() < epsilon:
                            a = envs.action_sample()
                        else:
                            a = agent.select_action(s)
                        s_ = dynamics_model.predict(s, a)
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

                min_reward = np.min(reward_episode_set)
                if min_reward < reward_threshold:
                    count = 0
                else:
                    count += 1
                if count >= converge_threshold:
                    steps_list.append(total_num_steps)
                    break

    res = np.mean(steps_list)
    if np.isnan(res):
        return INF
    return res


def objective(trial):
    n_train = trial.suggest_int('n', 1, 120)
    start_train = trial.suggest_int('start', -1, 19)
    h_train = trial.suggest_int('h', 1, 50)
    m_train = trial.suggest_int('m', 1, 50)
    return single_run(n_train, start_train, h_train, m_train)


if __name__ == "__main__":
    max_trials = 10000
    s = time.time()
    study = optuna.create_study(study_name='Search', direction='minimize')
    study.optimize(objective, n_trials=max_trials)
    print(f'Bets Parameters: {study.best_params}')
    e = time.time()
    print(f'Time Cost: {time.strftime("%Hh %Mm %Ss", time.gmtime(e - s))}')
    print()
