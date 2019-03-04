import tensorflow as tf
from baselines.her.util import store_args, nn
import numpy as np
from tensorflow.math import cos as tcos
from tensorflow.math import sin as tsin

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, env=None, n_arms=None, **kwargs):
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
        self.ag_tf = inputs_tf['ag']

        self.env = env.unwrapped

        # End effector value
        #end_eff = self.o_tf[...,self.env.o_ndx2:]
        
        # Prepare inputs for actor and critic.
        o = self.o_tf[...,:self.env.o_ndx] #self.o_stats.normalize(self.o_tf)[...,:self.env.o_ndx
        g = self.g_tf #- end_eff #self.g_stats.normalize(self.g_tf - end_eff)
        
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

	
def make_rot_matrix(t):
    cosines = tf.reshape(tcos(t), [-1, 1, 1])
    sines   = tf.reshape(tsin(t), [-1, 1, 1])

    r1 = tf.concat(axis=2, values=[cosines, -sines])
    r2 = tf.concat(axis=2, values=[sines, cosines])

    R = tf.concat(axis=1, values=[r1, r2])
    return R

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

class ActorCriticDiff:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 n_arms, learn_kin=False, conn_type='sums', env=None, **kwargs):
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
        # Access to the environment type 
        self.env = env.unwrapped
        
        # N-Arms
        self.n_arms = n_arms
        
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']
        self.ag_tf = inputs_tf['ag']

        # Raw observations
        #o_raw = self.o_tf[...,:self.env.o_ndx]
        # Joint poses

        # Normalization
        o = self.o_tf[...,:self.env.o_ndx] #self.o_stats.normalize(self.o_tf)[...,:self.env.o_ndx
        g = self.g_tf #self.g_stats.normalize(self.g_tf)

        # Expose for debugging
        self.o_normalized = o
        self.g_normalized = g

        # Reshape inputs for the number of arms
        u = narm_reshape(self.u_tf, n_arms)
        o = narm_reshape(o, n_arms)
        #g = narm_reshape(g, n_arms+1)
        
        # Outputs
        pi_tfs    = [None] * n_arms
        Q_pi_tfs  = [None] * n_arms
        Q_tfs     = [None] * n_arms
        
        # Jacobian vals
        joint_jacp = self.o_tf[...,self.env.o_ndx:self.env.o_ndx2]
        joint_jacp = narm_reshape(joint_jacp, n_arms)

        # End effector value
        #end_eff = self.o_tf[...,self.env.o_ndx2:]
        
        # Calculate the loss
        #L = tf.reduce_sum(tf.square(g-end_eff), axis=1, keepdims=True)
        # Compute the gradient of the loss using chain rule
        #grad_end_eff = tf.gradients(L, end_eff)[0]
        
        # This gradient makes the assumption of relative gradients!!
        grad_end_eff = -2*g # analytical gradient
        gradL = tf.matmul(joint_jacp, tf.reshape(grad_end_eff, (-1, dimg, 1)))
        
        ###########################
        # Solve a quadratic to get equal no of params
        ##########################
        l = self.layers
        dimo = self.dimo
        dimg = self.dimg
        dimu = self.dimu
        dimo2 = o[:,0].shape.as_list()[-1]
        dimg2 = 1 #grads[:,0].shape.as_list()[-1]
        dimu2 = u[:,0].shape.as_list()[-1]
        H = self.hidden
        n = n_arms
        hidden = solve_quadratic(l, dimo, dimg, dimu, dimo2, dimg2, dimu2, H, n)-1 
        
        for i in range(n_arms):
            o_i = o[:, i]
            u_i = u[:, i]
            
            # Differentiation chain for method
            #g_i = tf.reduce_sum(grad_end_eff * joint_jacp[:,i], 1, keepdims=True)
            g_i = gradL[:,i]

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

class ActorCriticAll:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, env=None, n_arms=None, **kwargs):
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
        self.ag_tf = inputs_tf['ag']

        self.env = env.unwrapped

        # End effector value
        end_eff = self.o_tf[...,self.env.o_ndx2:]
        
        # Prepare inputs for actor and critic.
        #o = self.o_stats.normalize(self.o_tf)[...,:self.env.o_ndx]
        #g = self.g_stats.normalize(self.g_tf)
        o = self.o_tf[...,:self.env.o_ndx]
        #g = self.g_tf 
        #o = self.o_tf[...,:self.env.o_ndx]
        
        # Jacobian vals
        # Calculate the loss
        joint_jacp = self.o_tf[...,self.env.o_ndx:self.env.o_ndx2]
        joint_jacp = narm_reshape(joint_jacp, n_arms)
        
        #L = tf.reduce_sum(tf.square(g_old-end_eff), axis=1, keepdims=True)
        # Compute the gradient of the loss using chain rule
        #grad_end_eff = tf.gradients(L, end_eff)[0]
        #gradL = tf.matmul(joint_jacp, tf.reshape(grad_end_eff, (-1, 3, 1)))
        #g = gradL[...,0]
        
        # Analytical gradient
        grad_end_eff = -2*self.g_tf
        gradL = tf.matmul(joint_jacp, tf.reshape(grad_end_eff, (-1, 3, 1)))

        g = gradL


        
        #self.g_diff = g - end_eff
        
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
        
        # Calculation of total params
        #tp = 2*(self.layers-1)*self.hidden*self.hidden + (2*(dimo + self.dimg + self.dimu + self.layers) + 1)*self.hidden + self.dimu + 1
