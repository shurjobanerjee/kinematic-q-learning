import tensorflow as tf
from baselines.her.util import store_args, nn
import numpy as np
from tensorflow.math import cos as tcos
from tensorflow.math import sin as tsin

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, env=None, n_arms=None, normalized=False, **kwargs):
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
        self.o_tf  = inputs_tf['o']
        self.g_tf  = inputs_tf['g']
        self.u_tf  = inputs_tf['u']
        
        # Un-linearize the observation
        self.env        = env.unwrapped
        o_normed        = self.o_stats.normalize(self.o_tf)
        obs_dict        = self.env.reshaper.unlinearize(self.o_tf) 
        obs_dict_normed = self.env.reshaper.unlinearize(o_normed) 

        # Prepare inputs for actor and critic.
        if not normalized:
            o = tf.layers.Flatten()(obs_dict['observation'])
            g = self.g_tf
        else:
            o = tf.layers.Flatten()(obs_dict_normed['observation'])
            g = self.g_stats.normalize(self.g_tf)
            #g = tf.stopgradient(g)
        
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
        

def narm_reshape(x, n_arm):
    return tf.reshape(x, [-1, n_arm, x.get_shape().as_list()[-1]//n_arm])

def calculate_loss(g):
    return tf.reduce_sum(tf.square(g))
	
class ActorCriticDiff:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 n_arms, learn_kin=False, conn_type='sums', env=None, normalized=False, **kwargs):
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
        self.o_tf  = inputs_tf['o']
        self.g_tf  = inputs_tf['g']
        self.u_tf  = inputs_tf['u']

        # Calculate the gradients of g prior to normalization
        loss = calculate_loss(self.g_tf)
        dl_dg = tf.gradients(loss, self.g_tf)
        import pdb; pdb.set_trace()
        
        # Access to the environment type 
        self.env = env.unwrapped
        
        # N-Arms
        self.n_arms = n_arms
        
        # Un-linearize the observation
        self.env = env.unwrapped
        
        # Reshape inputs for the number of arms
        u = narm_reshape(self.u_tf, n_arms)
        
        # Normalize the observations and gradients
        observations = self.o_tf
        if normalized:
            observations = self.o_stats.normalize(observations)
        
        # Extract observations specifics
        obs_dict = self.env.reshaper.unlinearize(observations) 
        o = obs_dict['observation']
        gradL = [obs_dict['jacp{}'.format(i)] for i in range(n_arms)]


        import pdb; pdb.set_trace()

        ########################################################################
        # Solve a quadratic to get equal no of params #FIXME not working right
        ########################################################################
        hidden = solve_quadratic(self.layers, self.dimo, self.dimg, self.dimu,
                o2=o[:,0].shape.as_list()[-1], g2=1,
                u2=u[:,0].shape.as_list()[-1], H=self.hidden, n=n_arms)-1 
        
        ########################################################################
        
        # Outputs
        pi_tfs    = [None] * n_arms
        Q_pi_tfs  = [None] * n_arms
        Q_tfs     = [None] * n_arms
        
        for i in range(n_arms):
            # Observataions and actions
            o_i = o[:, i]
            u_i = u[:, i]
            
            # Differentiation chain for method
            g_i = gradL[i]
            
            # Input Pi
            input_pis_i = tf.concat(axis=1, values=[o_i, g_i])
            
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
            #total_params()
            #tp = 2*(self.layers-1)*hidden*hidden + (2*(2 + 1 + 1 + self.layers) + 1)*hidden + (1 + 1)

        with tf.variable_scope('pi'):
            self.pi_tf = tf.concat(axis=1, values=pi_tfs)
        

        with tf.variable_scope('Q'):
            # for policy training
            self.Q_pi_tf = sum(Q_pi_tfs)
            self.Q_tf    = sum(Q_tfs)
        
        total_params()
        

def compute_end_eff_pos(l, o, g, n_arms):
    
    # End effector posiions
    ends = [None] * n_arms
    armrad = 0
    center_coord = 0
    for i in range(n_arms):
        armrad +=  o[:,i:i+1,0]
        armdx_dy = tf.concat(axis=1, values = [l[:,i:i+1] * tcos(armrad), 
                                               l[:,i:i+1] * tsin(armrad)])
        ends[i] = center_coord + armdx_dy
        center_coord = ends[i]

    return ends[-1]

def solve_quadratic(l, o, g, u, o2, g2, u2, H, n):
    a   = 2*(l-1)
    b   = 2*(o2+g2+u2+l) + 1
    c_1 = (u2+1)
    c_2 = - 2*(l-1)*H*H/n
    c_3 = -(2*(o+g+u)+1)*H/n
    c_4 = -(u+1)/n
    c   = c_1 + c_2 + c_3 + c_4
    hidden = (- b + np.sqrt(b**2-4*a*c))/(2*a)

    return int(hidden)

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

