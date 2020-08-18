from REINFORCE import REINFORCE
from collections import deque
from matplotlib import pyplot as plt
from kernel import kernel
import gym
import tensorflow as tf 
import numpy as np 
import sklearn
import sklearn.preprocessing
import DHMP

env = gym.make('MountainCarContinuous-v0')
env.seed(1)

state_dim   = env.observation_space.shape[0]
actions_dim = env.action_space.shape[0] 
action_bound = [env.action_space.low, env.action_space.high]

state_space_samples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)

def scale_state(state):                  
    scaled = scaler.transform([state])
    return scaled

sess = tf.Session()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)

def policy_network(states):

    layer1 = tf.layers.dense(
        inputs=states,
        # inputs=tf.nn.l2_normalize(self.tf_states),
        units=128,
        activation=tf.nn.tanh,  # relu activation
        kernel_initializer=tf.random_normal_initializer(),
        bias_initializer=tf.constant_initializer(0),
        name='fc1'
    )

    layer2 = tf.layers.dense(
        inputs=layer1,
        # inputs=tf.nn.l2_normalize(self.tf_states),
        units=128,
        activation=tf.nn.tanh,  # relu activation
        kernel_initializer=tf.random_normal_initializer(),
        bias_initializer=tf.constant_initializer(0),
        name='fc2'
    )

    mu = tf.layers.dense(
        inputs=layer2,
        # inputs=tf.nn.l2_normalize(self.tf_states),
        units=actions_dim,
        activation=tf.nn.tanh,  # relu activation
        kernel_initializer=tf.random_normal_initializer(),
        bias_initializer=tf.constant_initializer(0),
        name='out'
    )

    sigma = tf.layers.dense(
                inputs=layer2,
                units=actions_dim,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc4_sigma'
            )

    sigma = tf.nn.softplus(sigma) + 1e-5
    
    return mu, sigma

pg_reinforce = REINFORCE(session = sess, 
                        optimizer = optimizer,
                        policy_network = policy_network,
                        state_dim = state_dim,
                        actions_dim = actions_dim,
                        action_bound = action_bound,
                        #init_exp=0.5,         # initial exploration prob
                        #final_exp=0.0,        # final exploration prob
                        #anneal_steps=10000,   # N steps for annealing exploration
                        discount_factor=0.99, # discount future rewards
                        #reg_param=0.001,      # regularization constants
                        #max_gradient=5,       # max gradient norms
                        summary_writer=None,
                        summary_every=100)

theta = np.random.uniform(low=0, high = 1, size=state_dim+actions_dim+1+state_dim)
noise_prior = np.sqrt(theta[-state_dim:])
noise_prior = np.diagflat(noise_prior)
theta = theta[0:-state_dim]
keps = 1e-5
B = np.eye(state_dim)

s_test = []
y_test = []
SMSE = []
SMSE_POG = []
order = []

T = 200
T_test = 500


# get test data
s_test = np.zeros((T_test, state_dim+actions_dim))
y_test = np.zeros((T_test, state_dim))
state = env.reset()
for i in range(T_test):
    state = scale_state(state)
    state=np.squeeze(state)

    action = pg_reinforce.sampleAction(state)
    next_state, reward, done, _ = env.step(action)

    if next_state[0] >= 0.5:
        c_reward = 10
        #c_reward = reward
    elif next_state[0] > -0.4:
        c_reward = (1+next_state[0])**2 #+ (next_state[1])
        #c_reward = reward
    else:
        #c_reward = (1-next_state[0])**2
        c_reward = reward

    pg_reinforce.storeRollout(state, action, c_reward)

    if (i%10 == 9):
        pg_reinforce.updateModel()


    s_test[i] = np.concatenate([state, action])
    y_test[i] = next_state

    state = next_state

# mean-center
y_test = y_test - np.mean(y_test, axis=0)


#start training
pg_reinforce.cleanUp()
s = np.zeros((1, state_dim + actions_dim))
y = np.zeros((1, state_dim))
s_comp = np.zeros((1, state_dim + actions_dim))
y_comp = np.zeros((1, state_dim))

state = env.reset()
for i in range(T):
    #env.render()
    state = scale_state(state)
    state = np.squeeze(state)
    
    action = pg_reinforce.sampleAction(state)
    next_state, reward, done, _ = env.step(action)

    if next_state[0] >= 0.5:
        c_reward = 10
        #c_reward = reward
    elif next_state[0] > -0.4:
        c_reward = (1+next_state[0])**2 #+ (next_state[1])
        #c_reward = reward
    else:
        #c_reward = (1-next_state[0])**2
        c_reward = reward

    pg_reinforce.storeRollout(state, action, c_reward)

    if (i%10 == 9):
        pg_reinforce.updateModel()

    # Dense GP
    if i > 0:
        D = s[1:]
        yy = y[1:]
        yy = yy-np.mean(yy, axis=0)
        yy_flatten = yy.flatten()

        # get training label variance, prepared for SMSE
        if i==1:
            v = 1
        else:
            v = (yy - np.mean(yy, axis=0))**2
            v = np.sum(v, axis=1)
            v = np.sum(v)/yy.shape[0]


        mu_test_posterior = []
        sigma_test_posterior = []
        standard_error = []

        for j in range(T_test):

            if i == 1:
                KDD = kernel(theta, D, D, X_vector=True, Y_vector=True)
                KDD = np.kron(KDD, B)
                KxtestD = kernel(theta, D, s_test[j], X_vector=True, Y_vector=True)
                KxtestD = np.kron(KxtestD, B)
            elif i > 1:
                KDD = kernel(theta, D, D, X_vector=False, Y_vector=False)
                KDD = np.kron(KDD, B)
                KxtestD = kernel(theta, D, s_test[j], X_vector=False, Y_vector=True)
                KxtestD = np.kron(KxtestD, B)

            mu_pos = KxtestD.T.dot(np.linalg.inv(KDD+np.kron(noise_prior**2, np.eye(D.shape[0])))).dot(yy_flatten)
            mu_test_posterior.append(mu_pos)

            Ktest = kernel(theta, s_test[j], s_test[j], X_vector=True, Y_vector=True)
            Ktest = np.kron(Ktest, B)
            sigma_pos = Ktest - KxtestD.T.dot(np.linalg.inv(KDD+np.kron(noise_prior**2, np.eye(D.shape[0])))).dot(KxtestD) + noise_prior**2
            sigma_test_posterior.append(sigma_pos)

            error = np.sum((y_test[j]-mu_pos)**2)/v
            standard_error.append(error)

        # get SMSE
        mean_error = np.mean(np.array(standard_error))
        SMSE.append(mean_error)

        print('step {}'.format(i))
        print('SMSE: {}'.format(mean_error))


    # POG
    if i > 0:
        if i == 1:
            s_comp = s_comp[1:]
            y_comp = y_comp[1:]
        # mean-center the training data
        yy_comp = y_comp
        yy_comp = yy_comp - np.mean(yy_comp, axis=0)
        yy_comp_flatten = yy_comp.flatten()

        # get training variance
        if i==1:
            v_comp = 1
        else:
            v_comp = (yy_comp - np.mean(yy_comp, axis=0))**2
            v_comp = np.sum(v_comp, axis=1)
            v_comp = np.sum(v_comp)/yy_comp.shape[0]

        standard_error_comp = []

        for k in range(T_test):
            if i == 1:
                KDD_comp = kernel(theta, s_comp, s_comp, X_vector=True, Y_vector=True)
                KDD_comp = np.kron(KDD_comp, B)
                KxtestD_comp = kernel(theta, s_comp, s_test[k], X_vector=True, Y_vector=True)
                KxtestD_comp = np.kron(KxtestD_comp, B)
            else:
                KDD_comp = kernel(theta, s_comp, s_comp, X_vector=False, Y_vector=False)
                KDD_comp = np.kron(KDD_comp, B)
                KxtestD_comp = kernel(theta, s_comp, s_test[k], X_vector=False, Y_vector=True)
                KxtestD_comp = np.kron(KxtestD_comp, B)

            mu_pos_comp = KxtestD_comp.T.dot(np.linalg.inv(KDD_comp+np.kron(noise_prior**2, np.eye(s_comp.shape[0])))).dot(yy_comp_flatten)
            
            Ktest_comp = kernel(theta, s_test[k], s_test[k], X_vector=True, Y_vector=True)
            Ktest_comp = np.kron(Ktest_comp, B)
            sigma_pos_comp = Ktest_comp - KxtestD_comp.T.dot(np.linalg.inv(KDD_comp+np.kron(noise_prior**2, np.eye(s_comp.shape[0])))).dot(KxtestD_comp) + noise_prior**2

            err_comp = np.sum((y_test[k]-mu_pos_comp)**2)/v_comp
            standard_error_comp.append(err_comp)

        mean_error_comp = np.mean(np.array(standard_error_comp))
        SMSE_POG.append(mean_error_comp)

        print('POG SMSE: {}'.format(mean_error_comp))

        # DHMP step
        newx = np.concatenate([state, action])
        I = DHMP.dhmp(s_comp, y_comp, newx, theta, keps, noise_prior, xvec=False)
        s_comp = s_comp[I]
        y_comp = y_comp[I]
        m = len(I) + 1
        order.append(m)

    temp_s = np.concatenate([state, action]).reshape(1,-1)
    s = np.concatenate([s, temp_s], axis=0)
    s_comp = np.concatenate([s_comp, temp_s], axis=0)
    temp_y = next_state.reshape(1,-1)
    y = np.concatenate([y, temp_y], axis=0)
    y_comp = np.concatenate([y_comp, temp_y], axis=0)
    state = next_state


with open('POG_with_order7.csv', 'w') as ofile:
    for j in range(1,len(SMSE)):
        line = str(SMSE[j]) + ',' + str(SMSE_POG[j]) + ',' + str(order[j]) + '\n'
        ofile.write(line)

plt.plot(range(1,len(SMSE)), SMSE[1:], label = 'Dense PG')
plt.plot(range(1,len(SMSE_POG)), SMSE_POG[1:], label = 'POG')
plt.xlabel("Number of Trajectories")
plt.ylabel("SMSE")
plt.legend()
plt.title('Dense PG vs. POG')
plt.savefig("POG7.png")

plt.figure()
plt.plot(range(len(order)), order, label = 'POG')
plt.plot(range(len(order)), range(len(order)), label = 'Dense PG')
plt.xlabel("Number of Trajectories")
plt.ylabel("Model Order")
plt.legend()
plt.savefig("order7.png")


    


    
    










