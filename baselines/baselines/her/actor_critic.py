import tensorflow as tf
from baselines.her.util import store_args, nn
import numpy as np
#FIXME
from params import Params

def linearize(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:]])

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
        
        import pdb; pdb.set_trace()
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
        
        total_params()

def solve_quadratic(H, l, o, g, u, n, x):
    a = l
    b = o + u + l + 1
    c =  - H * ( l*(H+1) + o + u + 1) * (x / n)
    h = (-b + np.sqrt(b*b-4*a*c))/(2*a)
    #print(h)
    return int(h)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
                yield l[i:i + n]

def split(a, n):
    k, m = divmod(len(a), n)
    ret = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    ret = [r + [r[-1]+1] for r in ret if r]
    return ret

class ActorCriticParts:
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
        
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        
        o_l = self.o_tf.get_shape().as_list()[-1] 
        g_l = self.g_tf.get_shape().as_list()[-1]
        u_l = self.u_tf.get_shape().as_list()[-1]
        
        # N-Arms
        n_arms = Params.n_arms
        o_ndxs = list(split(list(range(o_l)), n_arms))
        g_ndxs = list(split(list(range(g_l)), n_arms-1))
        u_ndxs = list(split(list(range(u_l)), n_arms))

        # Observations are per arms
        if Params.etype == 'Arm':
            df = 5 #Params.n_arms - 1
        else:
            df = Params.n_arms - 1
        os        = [None] * df
        gs        = [None] * df
        input_pis = [None] * df
        u_tfs     = [None] * df
        pi_tfs    = [None] * df
        input_Q_1 = [None] * df
        input_Q_2 = [None] * df
        Q_pi_tfs  = [None] * df
        Q_tfs     = [None] * df

        
        for i in range(df):#len(g_ndxs)):
            if i < (df - 1):
                gs[i] = g[...,g_ndxs[i][0]:g_ndxs[i][-1]]
                os[i] = o[...,o_ndxs[i][0]:o_ndxs[i][-1]]
                u_tfs[i] = self.u_tf[...,u_ndxs[i][0]:u_ndxs[i][-1]]
            else:
                gs[i] = g[...,g_ndxs[i][0]:g_ndxs[i][-1]]
                #os[i] = o[...,o_ndxs[i][0]:o_ndxs[i+1][-1]]
                #u_tfs[i] = self.u_tf[...,u_ndxs[i][0]:u_ndxs[i+1][-1]]
                
                #gs[i] = g[...,g_ndxs[i][0]:]
                os[i] = o[...,o_ndxs[i][0]:]
                u_tfs[i] = self.u_tf[...,u_ndxs[i][0]:]
            
            # Network structure is the same once dim. is sorted out
            input_pis[i] = tf.concat(axis=1, values=[os[i], gs[i]])
            
            if i < (df - 1):
                hidden = solve_quadratic(self.hidden, self.layers, dimo, dimg, dimu, n_arms, 1)
                with tf.variable_scope('pi{}'.format(i)):
                    pi_tfs[i] = self.max_u * tf.tanh(nn(
                        input_pis[i], [hidden] * self.layers + [1]))
            else:
                u_last_dim = u_tfs[i].shape.as_list()[-1]
                hidden = solve_quadratic(self.hidden, self.layers, dimo, dimg, dimu, n_arms, u_last_dim)
                with tf.variable_scope('pi{}'.format(i)):
                    pi_tfs[i] = self.max_u * tf.tanh(nn(
                        input_pis[i], [hidden] * self.layers + [u_last_dim]))
            
            with tf.variable_scope('Q{}'.format(i)):
                # for policy training
                input_Q_1[i] = tf.concat(axis=1, values=[os[i], gs[i], pi_tfs[i] / self.max_u])
                Q_pi_tfs[i] = nn(input_Q_1[i], [hidden] * self.layers + [1])
                
                # for critic training
                input_Q_2[i] = tf.concat(axis=1, values=[os[i], gs[i], u_tfs[i] / self.max_u])
                Q_tfs[i] = nn(input_Q_2[i], [hidden] * self.layers + [1], reuse=True)
        
        with tf.variable_scope('pi'):
            self.pi_tf = tf.concat(axis=1, values=pi_tfs)
        
        with tf.variable_scope('Q'):
            # for policy training
            self.Q_pi_tf = sum(Q_pi_tfs)
            self.Q_tf    = sum(Q_tfs)
        
        total_params()
	

def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print("Total Params: {}".format(total_parameters))
