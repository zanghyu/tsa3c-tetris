from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
from constants import constants
#from tetris_fun_beta import ACTION_SIZE
#from tetris_fun_beta import OBSERVATION_SHAPE
from tetris_fun import ACTION_SIZE
from tetris_fun import OBSERVATION_SHAPE
import random
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

def discount(x, gamma):
    """
        x = [r1, r2, r3, ..., rN]
        returns [r1 + r2*gamma + r3*gamma^2 + ...,
                   r2 + r3*gamma + r4*gamma^2 + ...,
                     r3 + r4*gamma + r5*gamma^2 + ...,
                        ..., ..., rN]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0, clip=False):
    """
    Given a rollout, compute its returns and the advantage.
    """
    # collecting transitions
    batch_si = np.asarray(rollout.states)
    batch_fs = np.asarray(rollout.fallingstates)

    batch_a = np.asarray(rollout.actions)

    # collecting target for value network
    # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])  # bootstrapping
    if rollout.unsup:
        rewards_plus_v += np.asarray(rollout.bonuses + [0])
    if clip:
        rewards_plus_v[:-1] = np.clip(rewards_plus_v[:-1], -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    batch_r = discount(rewards_plus_v, gamma)[:-1]  # value network target

    # collecting target for policy network
    rewards = np.asarray(rollout.rewards)
    if clip:
        rewards = np.clip(rewards, -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    vpred_t = np.asarray(rollout.values + [rollout.r])
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    # Eq (10): delta_t = Rt + gamma*V_{t+1} - V_t
    # Eq (16): batch_adv_t = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]

    return Batch(batch_si, batch_fs, batch_a, batch_adv, batch_r, rollout.terminal, features, rollout.tot_cleared)

Batch = namedtuple("Batch", ["si", "fs", "a", "adv", "r", "terminal", "features", "tot_cleared"])

class PartialRollout(object):
    """
    A piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    
    """
    def __init__(self, unsup=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.fallingstates = []
        self.unsup = unsup
        self.tot_cleared = 0
        if self.unsup:
            self.bonuses = []
            self.end_state = None


    def add(self, state, fallingstate, action, reward, value, terminal, features, tot_cleared,
                bonus=None, end_state=None):
        self.states += [state]
        self.fallingstates += [fallingstate]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.tot_cleared = tot_cleared
        if self.unsup:
            self.bonuses += [bonus]
            self.end_state = end_state

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.fallingstates.extend(other.fallingstates)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.tot_cleared = other.tot_cleared
        self.features.extend(other.features)
        if self.unsup:
            self.bonuses.extend(other.bonuses)
            self.end_state = other.end_state

class RunnerThread(threading.Thread):
    """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """
    def __init__(self, gameState, policy, num_local_steps, predictor):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)  # ideally, should be 1. Mostly doesn't matter in our case.
        self.num_local_steps = num_local_steps
        self.gameState = gameState
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.predictor = predictor

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.gameState, self.policy, self.num_local_steps,
                                        self.summary_writer, self.predictor)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(gameState, policy, num_local_steps, summary_writer, predictor):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    gameState.reinit()
    last_state = gameState.get_state()
    last_features = policy.get_initial_features()  # reset lstm memory
    last_falling_state = last_state

    length = 0
    rewards = 0
    values = 0
    tot_cleared = 0
    avg_reward = 0
    step_cnt = 0
    terminal_end = False
    rollout = None
    if predictor is not None:
        ep_bonus = 0
        life_bonus = 0

    while True:
        terminal_end = False
        rollout = PartialRollout(predictor is not None)

        for _ in range(num_local_steps):
            # run policy
            fetched = policy.act(last_state, last_falling_state, *last_features)

        # policy.sample, policy.vf, policy.state_out
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            # run environment: get action_index from sampled one-hot 'action'

            stepAct = action.argmax()
            act = [0]*ACTION_SIZE
            act[stepAct] = 1
            state, falling_state, reward, terminal, cleared = gameState.frame_step(act)
            state = np.reshape(state, (10, 20, 1))
            state = np.append(state, last_state[:, :, 0:3], axis=2)
            falling_state = np.reshape(falling_state, (10, 20, 1))
            falling_state = np.append(falling_state, last_state[:, :, 0:3], axis=2)

            tot_cleared += cleared
            avg_reward += reward
            step_cnt += 1
            if step_cnt % 500 == 0:
                rollout_avg_reward = float(avg_reward) / float(500)
                avg_reward = 0
                step_cnt = 0
                print("500 steps avarage reward:{}".format(str(rollout_avg_reward)))

            curr_tuple = [last_state, last_falling_state, action, reward, value_, terminal, last_features, tot_cleared]

            # collect the experience
            rollout.add(*curr_tuple)
            rewards += reward
            length += 1
            values += value_[0]

            last_state = state
            last_falling_state = falling_state
            last_features = features

            if terminal:
                # prints summary of each life if envWrap==True else each game

                length = 0
                rewards = 0
                terminal_end = True
                last_features = policy.get_initial_features()  # reset lstm memory
                gameState.reinit()
                last_state = gameState.get_state()
                last_falling_state = last_state

            if terminal_end:
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, last_falling_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout


class A3C(object):
    def __init__(self, gameState, task):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.

        """
        self.task = task
        self.gameState = gameState
        predictor = None
        numaction = ACTION_SIZE
        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(OBSERVATION_SHAPE, numaction)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)


        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(OBSERVATION_SHAPE, numaction)
                pi.global_step = self.global_step


            # Computing a3c loss: https://arxiv.org/abs/1506.02438
            self.ac = tf.placeholder(tf.float32, [None, numaction], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")
            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)
            # 1) the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, 1) * self.adv)  # Eq (19)
            # 2) loss of value function: l2_loss = (x-y)^2/2
            vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.vf - self.r))  # Eq (28)
            # 3) entropy to ensure randomness
            entropy = - tf.reduce_mean(tf.reduce_sum(prob_tf * log_prob_tf, 1))
            # final a3c loss: lr of critic is half of actor
            self.loss = pi_loss + 0.5 * vf_loss - entropy * constants['ENTROPY_BETA']

            # compute gradients
            grads = tf.gradients(self.loss * 20.0, pi.var_list)  # batchsize=20. Factored out to make hyperparams not depend on it.

            # computing predictor loss


            self.runner = RunnerThread(gameState, pi, constants['ROLLOUT_MAXLEN'],
                                        predictor)

            # storing summaries
            bs = tf.to_float(tf.shape(pi.x)[0])
            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss)
                tf.summary.scalar("model/value_loss", vf_loss)
                tf.summary.scalar("model/entropy", entropy)
                tf.summary.image("model/state", pi.x)  # max_outputs=10
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))

                self.summary_op = tf.summary.merge_all()
            else:
                tf.scalar_summary("model/policy_loss", pi_loss)
                tf.scalar_summary("model/value_loss", vf_loss)
                tf.scalar_summary("model/entropy", entropy)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))

                self.summary_op = tf.merge_all_summaries()

            # clip gradients
            grads, _ = tf.clip_by_global_norm(grads, constants['GRAD_NORM_CLIP'])
            grads_and_vars = list(zip(grads, self.network.var_list))

            # update global step by batch size
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            # TODO: make optimizer global shared, if needed
            print("Optimizer: RMSProp with lr: %f" % (constants['LEARNING_RATE']))
            print("Input observation shape: ",OBSERVATION_SHAPE)
            opt = tf.train.RMSPropOptimizer(constants['LEARNING_RATE'])
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

            # copy weights from the parameter server to the local model
            sync_var_list = [v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)]

            self.sync = tf.group(*sync_var_list)

            # initialize extras
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
        Take a rollout from the queue of the thread runner.
        """
        # get top rollout from queue (FIFO)
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                # Now, get remaining *available* rollouts from queue and append them into
                # the same one above. If queue.Queue(5): len=5 and everything is
                # superfast (not usually the case), then all 5 will be returned and
                # exception is raised. In such a case, effective batch_size would become
                # constants['ROLLOUT_MAXLEN'] * queue_maxlen(5). But it is almost never the
                # case, i.e., collecting  a rollout of length=ROLLOUT_MAXLEN takes more time
                # than get(). So, there are no more available rollouts in queue usually and
                # exception gets always raised. Hence, one should keep queue_maxlen = 1 ideally.
                # Also note that the next rollout generation gets invoked automatically because
                # its a thread which is always running using 'yield' at end of generation process.
                # To conclude, effective batch_size = constants['ROLLOUT_MAXLEN']
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
        Process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=constants['GAMMA'], lambda_=constants['LAMBDA'], clip=True)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.local_network.y: batch.fs,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
        }


        fetched = sess.run(fetches, feed_dict=feed_dict)
        if batch.terminal:
            print("Global Step Counter: %d"%fetched[-1])
        if batch.tot_cleared > 2:
            print("Global Step Counter: {}, cleared: {}".format(str(fetched[-1]),str(batch.tot_cleared)))

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
