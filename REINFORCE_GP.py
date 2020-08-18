from scipy.stats import multivariate_normal
import tensorflow as tf 
import numpy as np 
import sklearn.preprocessing

SEED = 7
np.random.seed(SEED)
tf.set_random_seed(SEED)

class REINFORCE(object):

    def __init__(self,  state_dim,
                        actions_dim,
                        action_bound,
                        alpha,
                        learning_rate,
                        l_max,
                        l_min,
                        N_max,
                        sigma,
                        layer,
                        discount_factor,
                        scale_reward,
                        mean): 

        self.state_dim         = state_dim
        self.actions_dim       = actions_dim
        self.action_bound      = action_bound
        self.discount_factor   = discount_factor
        self.alpha             = alpha
        self.learning_rate     = learning_rate
        #self.learning_rate     = tf.placeholder(tf.float32, shape=[])
        self.l_max             = l_max
        self.l_min             = l_min
        self.N_max             = N_max
        self.sigma             = sigma
        self.layer             = layer
        self.scale_reward      = scale_reward
        self.mean              = mean

        self.state_buffer      = []
        self.reward_buffer     = []
        self.action_buffer     = []
        self.GP_product_buffer = []
        self.Q_buffer          = []
        self.rho_buffer        = []
        self.pi_prob_buffer    = []
        self.log_prob_buffer   = []

        self.optimizer         = tf.train.AdamOptimizer(self.learning_rate)
        self.build_net()
        self.session           = tf.Session()

        self.session.run(tf.global_variables_initializer())

        self.saver= tf.train.Saver()

    def build_net(self):
        with tf.name_scope("model_inputs"):
            # raw state representation
            self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
            self.taken_actions = tf.placeholder(tf.float32, (None, self.actions_dim), name="taken_actions")
            self.discounted_rewards = tf.placeholder(tf.float32, (None, 1), name="discounted_rewards")
            self.co = tf.placeholder(tf.float32, (None, 1), name="coefficients")

        with tf.variable_scope('REINFORCE/layers'):
            layer1 = tf.layers.dense(
                inputs=self.states,
                # inputs=tf.nn.l2_normalize(self.tf_states),
                units=self.layer,
                activation=tf.nn.tanh,  # relu activation
                kernel_initializer=tf.random_normal_initializer(mean=0.05,stddev=0.5,seed=SEED),
                bias_initializer=tf.constant_initializer(0),
                name='fc1'
            )

            layer2 = tf.layers.dense(
                inputs=layer1,
                # inputs=tf.nn.l2_normalize(self.tf_states),
                units=self.layer,
                activation=tf.nn.tanh,  # relu activation
                kernel_initializer=tf.random_normal_initializer(mean=0.05,stddev=0.5,seed=SEED),
                bias_initializer=tf.constant_initializer(0),
                name='fc2'
            )

            self.mu = tf.layers.dense(
                inputs=layer2,
                # inputs=tf.nn.l2_normalize(self.tf_states),
                units=self.actions_dim,
                activation=tf.nn.tanh,  # relu activation
                kernel_initializer=tf.random_normal_initializer(mean=0.05,stddev=0.5,seed=SEED),
                bias_initializer=tf.constant_initializer(0),
                name='out'
            )

            if self.sigma is None:
                self.sigma = tf.layers.dense(
                    inputs=layer2,
                    units=self.actions_dim,
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.random_normal_initializer(mean=0.05,stddev=0.5,seed=SEED),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='fc4_sigma'
                )
                self.sigma = tf.nn.softplus(self.sigma) + 1.0e-5

        normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.predicted_actions =  tf.placeholder(tf.float32, (None, self.actions_dim), name="predicted_actions")
        self.predicted_actions = tf.clip_by_value(tf.squeeze(normal_dist.sample(1)), *self.action_bound)

        with tf.name_scope("compute_prob"):
            self.prob = normal_dist.prob(self.taken_actions)

        with tf.name_scope('REINFORCE/loss'):
            self.log_probs = normal_dist.log_prob(self.taken_actions)
            if self.mean:
                self.loss = -tf.reduce_mean(self.co * self.log_probs * self.discounted_rewards)
            else:
                self.loss = -tf.reduce_sum(self.co * self.log_probs * self.discounted_rewards)
            #self.loss = tf.reduce_mean(-self.co * tf.log(tf.sqrt(1/(2 * np.pi * self.sigma**2)) * tf.exp(-(self.taken_actions - self.mu)**2/(2 * self.sigma**2))) * self.discounted_rewards)

        with tf.name_scope("REINFORCE/compute_gradients"):   
            # compute gradients
            #self.gradients = self.optimizer.compute_gradients(-self.loss)
            self.op = self.optimizer.minimize(self.loss)

        #with tf.name_scope("REINFORCE/train"):
            # apply gradients to update policy network
            #self.train_op = self.optimizer.apply_gradients(self.gradients)

    def cleanUp(self):
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []
        self.discount_reward_buffer = []
        self.GP_product_buffer = []
        self.Q_buffer = []
        self.rho_buffer = []
        self.pi_prob_buffer = []

    def storeRollout(self, state, action, reward):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)
        # if (len(self.Q_buffer)==0):
        #     self.Q_buffer.append(reward)
        # else:
        #     n = len(self.Q_buffer)
        #     q = self.Q_buffer[-1] + self.discount_factor**n * reward

    def storeRollout_GP(self, state, action, reward):
        self.storeRollout(state, action, reward)

        p = self.session.run(self.prob, feed_dict={
            self.states: np.array(state).reshape(-1,self.state_dim),
            self.taken_actions: np.array(action).reshape(-1,self.actions_dim)
        })
        self.pi_prob_buffer.append(p)

    def compute_GP_state_occu(self, state, action, reward, GP_mu, GP_sigma):
        self.storeRollout_GP(state, action, reward)

        GP_dist = multivariate_normal(mean=GP_mu, cov=GP_sigma)
        gp_prob = GP_dist.pdf(state)

        if len(self.GP_product_buffer) == 0:
            self.GP_product_buffer.append(1)
            self.rho_buffer.append(self.GP_product_buffer[-1])
        else:
            length= len(self.GP_product_buffer)
            v = self.GP_product_buffer[-1] * gp_prob
            self.GP_product_buffer.append(v)
            vv = self.rho_buffer[-1] + self.discount_factor**(length) * v
            self.rho_buffer.append(vv)

    def computeDiscountReward(self):
        N = len(self.reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self.reward_buffer[t] + self.discount_factor * r
            #discounted_rewards[t] = r * (self.discount_factor**t)
            discounted_rewards[t] = r

        if self.scale_reward:
            discounted_rewards = sklearn.preprocessing.scale(discounted_rewards)
            # discounted_rewards -= np.mean(discounted_rewards)
            # discounted_rewards /= np.std(discounted_rewards)

        self.discount_reward_buffer = discounted_rewards

    def sampleAction(self, states):
        states = states.reshape(-1,self.state_dim)   # single state
        action = self.session.run(self.predicted_actions, {self.states: states})  # single action
        return action

    def _gen_learning_rate(self, iteration):
        if iteration > self.N_max:
            return self.l_min
        alpha = 2 * self.l_max
        beta = np.log((alpha / self.l_min - 1)) / self.N_max
        return alpha / (1 + np.exp(beta * iteration))

    # def minibatches(self, samples, batch_size):
    #     for i in range(0, len(samples), batch_size):
    #         yield samples[i:i + batch_size]

    def GP_update(self, iteration):
        self.rho_buffer = (1-self.discount_factor)*np.array(self.rho_buffer).reshape(-1,1)
        self.computeDiscountReward()
        N= len(self.action_buffer)
        #lr = self._gen_learning_rate(iteration)

        co=np.zeros(N)
        for i in range(N):
            co[i] = 1/(1-self.discount_factor) * self.rho_buffer[i] * self.pi_prob_buffer[i]
        gamma_buffer = np.logspace(start=0,stop=N-2,num=N-1,base=self.discount_factor)

        coeff = np.zeros(N)
        if self.mean:
            coeff[0] = (1-self.alpha) * co[0] * (N-1)
        else:
            coeff[0] = (1-self.alpha) * co[0] * np.sum(gamma_buffer)
        # coeff[0] = (1-self.alpha) * co[0] * (N-1)
        for i in range(1,N-1):
            if self.mean:
                coeff[i] = self.alpha + (1-self.alpha)*co[i]*(N-1-i)
            else:
                coeff[i] = (self.discount_factor**(i-1))*self.alpha + (1-self.alpha)*co[i]*np.sum(gamma_buffer[i:])
            # coeff[i] = self.alpha + (1-self.alpha)*co[i]*(N-1-i)
        if self.mean:
            coeff[N-1] = self.alpha 
        else:
            coeff[N-1] = self.alpha * self.discount_factor**(N-2)
        # coeff[N-1] = self.alpha 
        self.session.run(self.op, feed_dict={
            self.states: np.array(self.state_buffer).reshape(-1,self.state_dim),
            self.taken_actions: np.array(self.action_buffer).reshape(-1,self.actions_dim),
            self.discounted_rewards: np.array(self.discount_reward_buffer).reshape(-1,1),
            self.co: coeff.reshape(-1,1),
            #self.learning_rate: lr
        })

        self.cleanUp()

    def save_model(self, name):
        save_path= self.saver.save(self.session, name)







            
        



