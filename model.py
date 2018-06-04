from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from constants import constants


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def cosineLoss(A, B, name):
    ''' A, B : (BatchSize, d) '''
    dotprod = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A,1), tf.nn.l2_normalize(B,1)), 1)
    loss = 1-tf.reduce_mean(dotprod, name=name)
    return loss


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)




def universeHeadx(x, nConvs=4):
    ''' universe agent example
        input: [None, 42, 42, 1]; output: [None, 288];
    '''
    print('Using universe head design')
    x = tf.nn.elu(conv2d(x, 32, "lx1", [5, 5], [2, 2]))
    x = tf.nn.elu(conv2d(x, 64, "lx2", [5, 5], [2, 2]))
    print(x)
    x = flatten(x)
    return x

def universeHeady(y, nConvs=4):
    ''' universe agent example
        input: [None, 42, 42, 1]; output: [None, 288];
    '''
    print('Using universe head design')
    y = tf.nn.elu(conv2d(y, 32, "ly1", [5, 5], [2, 2]))
    y = tf.nn.elu(conv2d(y, 64, "ly2", [5, 5], [2, 2]))
    print(y)
    y = flatten(y)
    return y


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, designHead='universe'):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x')
        self.y = y = tf.placeholder(tf.float32, [None] + list(ob_space), name='y')
        size = 256

        x = universeHeadx(x)
        y = universeHeady(y)

        mlpsize = x.get_shape()[1]
        mlp = tf.concat(1,[x,y])

        mlp = tf.nn.relu(linear(mlp, mlpsize, "li1", normalized_columns_initializer(0.01)))
        mlp = tf.nn.relu(linear(mlp, mlpsize, "li2", normalized_columns_initializer(0.01)))

        mlp = tf.expand_dims(mlp, [0])
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(mlp)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, mlp, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        mlp = tf.reshape(lstm_outputs, [-1, size])

        self.vf = tf.reshape(linear(mlp, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # [0, :] means pick action of first state from batch. Hardcoded b/c
        # batch=1 during rollout collection. Its not used during batch training.
        self.logits = linear(mlp, ac_space, "action", normalized_columns_initializer(0.01))
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)



    def get_initial_features(self):
        # Call this function to get reseted lstm memory cells
        return self.state_init

    def act(self, ob, fob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.y: [fob], self.state_in[0]: c, self.state_in[1]: h})

    def act_inference(self, ob, fob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.y: [fob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, fob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.y: [fob], self.state_in[0]: c, self.state_in[1]: h})[0]


