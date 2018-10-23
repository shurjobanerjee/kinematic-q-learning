import tensorflow as tf
from baselines.her.util import store_args, nn
import numpy as np

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)


def narm_reshape(x, n_arm):
    return tf.reshape(x, [-1, n_arm, x.get_shape().as_list()[-1]//n_arm])

def solve_quadratic(H, l, o, g, u, n, x):
    a = l
    b = o + u + l + 1
    c =  - H * ( l*(H+1) + o + u + 1) * (x / n)
    h = (-b + np.sqrt(b*b-4*a*c))/(2*a)
    #print(h)
    return int(h)
	
class ActorCriticParts:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 n_arms, learn_kin=False, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        # N-Arms
        self.n_arms = n_arms
        
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']
        
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        # Expose for debugging
        self.o_normalized  = o
        self.g_normalized = g

        # Reshape inputs for the number of arms
        u = narm_reshape(self.u_tf, n_arms)
        o = narm_reshape(o, n_arms)
        
        # Outputs
        pi_tfs    = [None] * n_arms
        Q_pi_tfs  = [None] * n_arms
        Q_tfs     = [None] * n_arms
        
        for i in range(n_arms):
            o_i = o[:, i]
            u_i = u[:, i]
            g_i = g
            
            input_pis_i = tf.concat(axis=1, values=[o_i, g_i])
            
            hidden = solve_quadratic(self.hidden, self.layers, dimo, dimg, dimu, n_arms, 1)
            with tf.variable_scope('pi{}'.format(i)):
                pi_tfs[i] = self.max_u * tf.tanh(nn(
                    input_pis_i, [hidden] * self.layers + [1]))

            with tf.variable_scope('Q{}'.format(i)):
                # for policy training
                input_Q_1_i = tf.concat(axis=1, values=[o_i, g_i, pi_tfs[i] / self.max_u])
                Q_pi_tfs[i] = nn(input_Q_1_i, [hidden] * self.layers + [1])
                
                # for critic training
                input_Q_2_i = tf.concat(axis=1, values=[o_i, g_i, u_i / self.max_u])
                Q_tfs[i] = nn(input_Q_2_i, [hidden] * self.layers + [1], reuse=True)

        with tf.variable_scope('pi'):
            self.pi_tf = tf.concat(axis=1, values=pi_tfs)
        
        with tf.variable_scope('Q'):
            # for policy training
            #self.Q_pi_tf = sum(Q_pi_tfs)
            #self.Q_tf    = sum(Q_tfs)

            Q1 = tf.concat(axis=1, values=Q_pi_tfs)
            self.Q_pi_tf = nn(Q1, [hidden] * self.layers + [1], reuse=tf.AUTO_REUSE)
            
            Q2 = tf.concat(axis=1, values=Q_tfs)
            self.Q_tf   = nn(Q2, [hidden] * self.layers + [1], reuse=tf.AUTO_REUSE)
