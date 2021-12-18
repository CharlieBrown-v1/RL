import numpy as np

from abc import abstractmethod
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


row = 0
col = 1
state = 2


class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


class myQAgent(QAgent):
    def __init__(self, width, height, state_count):
        QAgent.__init__(self, )
        self.width = width
        self.height = height
        self.policy = np.zeros(width * height * state_count).astype(int)

    def obs2index(self, obs):
        size = self.width * self.height
        return int(obs[state] * size + obs[row] * self.width + obs[col])

    def index2obs(self, index):
        size = self.width * self.height
        obs = np.zeros(3, dtype=np.int)
        obs[state] = index // size
        index -= obs[state] * size
        obs[row] = index // self.width
        obs[col] = index % self.width
        return obs

    @abstractmethod
    def select_action(self, obs):
        index = self.obs2index(obs)
        return self.policy[index]

    def update(self, obs, action):
        index = self.obs2index(obs)
        self.policy[index] = action


class Model:
    def __init__(self, width, height, policy):
        self.width = width
        self.height = height
        self.policy = policy
        pass

    @abstractmethod
    def store_transition(self, s, a, r, s_):
        pass

    @abstractmethod
    def sample_state(self):
        pass

    @abstractmethod
    def sample_action(self, s):
        pass

    @abstractmethod
    def predict(self, s, a):
        pass


class DynaModel(Model):
    def __init__(self, width, height, state_count, action_shape, policy: myQAgent):
        Model.__init__(self, width, height, policy)
        self.model = np.zeros((width * height * state_count, action_shape))
        self.buffer = []
        self.state_action_list = [[] for _ in np.arange(width * height * state_count)]

    def store_transition(self, s, a, r, s_):
        state_index = self.policy.obs2index(s)
        state_index_next = self.policy.obs2index(s_)
        self.model[state_index, a] = state_index_next
        self.buffer.append(s)
        if a not in self.state_action_list[state_index]:
            self.state_action_list[state_index].append(a)

    def sample_state(self):
        buffer_index = np.random.randint(0, len(self.buffer))
        s = self.buffer[buffer_index]
        return s, buffer_index

    def sample_action(self, s):
        state_index = self.policy.obs2index(s)
        assert 1 <= len(self.state_action_list[state_index]) <= 4
        action = np.random.choice(self.state_action_list[state_index])
        return action

    def predict(self, s, a):
        state_index = self.policy.obs2index(s)
        index = self.model[state_index, a]
        s_ = self.policy.index2obs(index)
        return s_

    def train_transition(self):
        pass


class NetworkModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
        self.x_next_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x_next')
        self.a_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='a')
        self.r_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        h1 = tf.layers.dense(tf.concat([self.x_ph, self.a_ph], axis=-1), units=256, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, units=256, activation=tf.nn.relu)
        self.next_x = tf.layers.dense(h2, units=3, activation=tf.nn.tanh) * 1.3 + self.x_ph
        self.x_mse = tf.reduce_mean(tf.square(self.next_x - self.x_next_ph))
        self.opt_x = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(self.x_mse)
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.variables_initializer(tf.global_variables()))
        self.buffer = []
        self.sensitive_index = []

    def norm_s(self, s):
        return s

    def de_norm_s(self, s):
        s = np.clip(np.round(s), 0, self.width - 1).astype(np.int32)
        s[2] = np.clip(s[2], 0, 1).astype(np.int32)
        return s

    def store_transition(self, s, a, r, s_):
        s = self.norm_s(s)
        s_ = self.norm_s(s_)
        self.buffer.append([s, a, r, s_])
        if s[-1] - s_[-1] != 0:
            self.sensitive_index.append(len(self.buffer) - 1)

    def train_transition(self, batch_size):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            s, a, r, s_ = self.buffer[idx]
            s_list.append(s)
            a_list.append([a])
            r_list.append(r)
            s_next_list.append(s_)

        x_mse = self.sess.run([self.x_mse,  self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]
        return x_mse

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        s, a, r, s_ = self.buffer[idx]
        return self.de_norm_s(s), idx

    def sample_action(self, s):
        return self.policy.select_action(s)

    def predict(self, s, a):
        s_ = self.sess.run(self.next_x, feed_dict={self.x_ph: [s], self.a_ph: [[a]]})
        return self.de_norm_s(s_[0])
