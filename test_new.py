from REINFORCE_GP import REINFORCE
from collections import deque
from matplotlib import pyplot as plt
from kernel import kernel
from scipy.stats import multivariate_normal
import tensorflow as tf 
import numpy as np 
import DHMP
import itertools
import sklearn
import sklearn.preprocessing
from continuous_mountain_car import Continuous_MountainCarEnv

env= Continuous_MountainCarEnv()
env.seed(1)

state_dim   = env.observation_space.shape[0]
actions_dim = env.action_space.shape[0] 
action_bound = [env.action_space.low, env.action_space.high]

state_space_samples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)

def scale_state(state):                  #requires input shape=(2,)
    scaled = scaler.transform([state])
    return np.squeeze(scaled)

alpha = 1
pg_reinforce = REINFORCE(state_dim = state_dim,
                            actions_dim = actions_dim,
                            action_bound = action_bound,
                            alpha=alpha,
                            learning_rate=1e-3,
                            l_max=1e-2,
                            l_min=1e-5,
                            N_max=2000,
                            sigma=None,
                            layer=128,
                            discount_factor=0.99,
                            scale_reward=False,
                            mean=True
                        )

render = True
scale = True

theta = np.random.uniform(low=0, high = 1, size=state_dim+actions_dim+1+state_dim)
noise_prior = np.sqrt(theta[-state_dim:])
noise_prior = np.diagflat(noise_prior)
theta = theta[0:-state_dim]
keps = 1.5e-4
B = np.eye(state_dim)

max_episode = 6000
if alpha < 1:
    max_episode = 3000
max_step = 300
eval_step = 300
num_test_per_episode = 3
solved = 0
solve_ep = 0

reward_his = []
window_reward_his = deque(maxlen=100)
window_mean = []

mean_test_reward = []
window_test = deque(maxlen=100)
test_window_mean = []

order = []

D = np.zeros((1, state_dim + actions_dim))
y = np.zeros((1, state_dim))

for i_episode in range(max_episode):
    pg_reinforce.cleanUp()
    state = env.init() #s_{0}
    if scale:
        state = scale_state(state)
    if render:
        env.render()
    total_rewards = 0
    next_state_error= []
    action = pg_reinforce.sampleAction(state) #a_{0}
    next_state, reward, done, _ = env.step(action)
    
    if next_state[0] >= 0.45:
        c_reward = 100
    # else:
    #     c_reward = 10*(next_state[0]+0.5)
    #     # c_reward = 1*(next_state[0]+1.2)**2
    elif next_state[0] > -0.5:
        c_reward = (1+next_state[0])**2 #+ (next_state[1])
        # c_reward = reward
    elif next_state[0] <= -0.5:
        c_reward = abs(next_state[1])
        c_reward = reward
    total_rewards += reward

    newx = np.concatenate([state, action])
    if i_episode==0:
        D = np.concatenate([D, newx.reshape(1,-1)], axis=0)
        y = np.concatenate([y, next_state.reshape(1,-1)], axis=0)
        D = D[1:]
        y = y[1:]

    if D.shape[0] == 1:
        xvec = True
    else:
        xvec = False
    
    yy_flatten = y.flatten()
    KDD = kernel(theta, D, D, X_vector=xvec, Y_vector=xvec)
    KDD = np.kron(KDD, B)
    KxtestD = kernel(theta, D, newx, X_vector=xvec, Y_vector=True)
    KxtestD = np.kron(KxtestD, B)
    #s_{1}
    mu_pos = KxtestD.T.dot(np.linalg.inv(KDD+np.kron(noise_prior**2, np.eye(D.shape[0])))).dot(yy_flatten) 

    Ktest = kernel(theta, newx, newx, X_vector=True, Y_vector=True)
    Ktest = np.kron(Ktest, B)
    sigma_pos = Ktest - KxtestD.T.dot(np.linalg.inv(KDD+np.kron(noise_prior**2, np.eye(D.shape[0])))).dot(KxtestD) + noise_prior**2

    if alpha < 1:
        I = DHMP.dhmp(D, y, newx, theta, keps, noise_prior, xvec=xvec)
        D = D[I]
        y = y[I]
        if i_episode != 0:
            D = np.concatenate([D, newx.reshape(1,-1)], axis=0)
            y = np.concatenate([y, next_state.reshape(1,-1)], axis=0)

    pg_reinforce.compute_GP_state_occu(state=state, action=action, reward=reward, GP_mu=mu_pos, GP_sigma=sigma_pos)

    for t in range(1, max_step):
        state= next_state
        if scale:
            state = scale_state(state)
        action = pg_reinforce.sampleAction(state)
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        if next_state[0] >= 0.45:
            c_reward = 100
        # else:
        #     #c_reward = 1*(next_state[0]+1.2)**2
        #     c_reward = 10*(next_state[0]+0.5)
        elif next_state[0] > -0.5:
            c_reward = (1+next_state[0])**2 #+ (next_state[1])
            #c_reward = reward
        elif next_state[0] <= -0.5:
            c_reward = abs(next_state[1])
            c_reward = reward
        total_rewards += reward

        pg_reinforce.compute_GP_state_occu(state=state, action=action, reward=reward, GP_mu=mu_pos, GP_sigma=sigma_pos)

        # calculate GP-based next state
        if alpha < 1:
            newx = np.concatenate([state, action])
            if D.shape[0] == 1:
                xvec = True
            else:
                xvec = False
            # yy = y - np.mean(y, axis=0) #mean center 
            # yy_flatten = yy.flatten()
            yy_flatten = y.flatten()
            KDD = kernel(theta, D, D, X_vector=xvec, Y_vector=xvec)
            KDD = np.kron(KDD, B)
            KxtestD = kernel(theta, D, newx, X_vector=xvec, Y_vector=True)
            KxtestD = np.kron(KxtestD, B)
            mu_pos = KxtestD.T.dot(np.linalg.inv(KDD+np.kron(noise_prior**2, np.eye(D.shape[0])))).dot(yy_flatten)

            Ktest = kernel(theta, newx, newx, X_vector=True, Y_vector=True)
            Ktest = np.kron(Ktest, B)
            sigma_pos = Ktest - KxtestD.T.dot(np.linalg.inv(KDD+np.kron(noise_prior**2, np.eye(D.shape[0])))).dot(KxtestD) + noise_prior**2

            I = DHMP.dhmp(D, y, newx, theta, keps, noise_prior, xvec=xvec)
            D = D[I]
            y = y[I]
            D = np.concatenate([D, newx.reshape(1,-1)], axis=0)
            y = np.concatenate([y, next_state.reshape(1,-1)], axis=0)

            error= np.sum(next_state-mu_pos)**2
            o= D.shape[0]
            print("episode: {}, time: {}, error: {}, order: {}".format(i_episode, t, error, o))
            next_state_error.append(error)

        if done:
            break

    reward_his.append(total_rewards)
    window_reward_his.append(total_rewards)
    window_mean.append(np.mean(window_reward_his))

    print("-"*30)
    pg_reinforce.GP_update(iteration=i_episode)
    pg_reinforce.cleanUp()

    # # eval current policy
    mean_test = []
    for i_test in range(num_test_per_episode):
        state = env.reset()
        eval_rewards = 0
        for i in range(eval_step):
            if scale:
                state = scale_state(state)
            action = pg_reinforce.sampleAction(state)
            next_state, reward, done, _ = env.step(action)
            eval_rewards += reward

            state = next_state
            state = np.array(state)
            if done: break

        mean_test.append(eval_rewards)
    
    mean_test = np.mean(mean_test)
    mean_test_reward.append(mean_test)
    window_test.append(mean_test)
    test_window_mean.append(np.mean(window_test))

    print("Episode {}".format(i_episode))
    if alpha < 1:
        print("State prediction error is: {}".format(np.mean(next_state_error)))
    print("Reward for this episode: {}".format(total_rewards))
    print("Average reward for last 100 episodes: {:.2f}".format(np.mean(window_reward_his)))
    print("Mean Test reward with current policy: {}".format(mean_test))
    print("Mean reward for last 100 test: {}".format(np.mean(window_test)))
    print("Model Order: {}".format(D.shape[0]))
    if mean_test >= 90 and solved == 0:
        solve_ep = i_episode
        solved = 1 

    if solved == 1:
        if np.mean(window_test) > 90:
            break
        if solve_ep <= 100 and i_episode>=100+solve_ep:
            break
        elif solve_ep > 100 and i_episode>=2*solve_ep:
            break
        
    print(solved)
    print(solve_ep)
    print("*"*30)

with open('reinforce_lr=8e-3_tanh_mean_2.csv', 'w') as ofile:
    for j in range(len(window_mean)):
        line = str(reward_his[j]) + ',' + str(window_mean[j]) + ',' + str(mean_test_reward[j]) + ',' + str(test_window_mean[j]) + '\n'
        ofile.write(line)

print("Finish training policy...")
pg_reinforce.save_model(name="reinforce_2.ckpt")









































