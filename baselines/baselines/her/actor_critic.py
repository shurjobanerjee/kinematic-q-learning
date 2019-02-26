import tensorflow as tf
from baselines.her.util import store_args, nn
import numpy as np
from tensorflow.math import cos as tcos
from tensorflow.math import sin as tsin

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, env=None,
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

        self.env = env.unwrapped

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)[...,:self.env.o_ndx]
        g = self.g_stats.normalize(self.g_tf)
        #o = self.o_tf[...,:self.env.o_ndx] #self.o_stats.normalize(self.o_tf)[...,:self.env.o_ndx]
        #g = self.g_tf #self.g_stats.normalize(self.g_tf)
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
        # Access to the environment type (
        self.env = env.unwrapped
        
        # N-Arms
        self.n_arms = n_arms
        
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Raw observations
        #o_raw = self.o_tf[...,:self.env.o_ndx]
        # Joint poses
        joint_poses = self.o_tf[...,self.env.o_ndx:]
        joint_poses = narm_reshape(joint_poses, n_arms)

        # Normalization
        o = self.o_stats.normalize(self.o_tf)[...,:self.env.o_ndx]
        #g = self.g_stats.normalize(self.g_tf)
        g = self.g_tf

        # Expose for debugging
        self.o_normalized = o
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
            g_i = g - joint_poses[:, i]
            
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
            if conn_type in ['sums', 'random']:
                self.Q_pi_tf = sum(Q_pi_tfs)
                self.Q_tf    = sum(Q_tfs)
            
            elif conn_type == 'fc':
                Q1 = tf.concat(axis=1, values=Q_pi_tfs)
                self.Q_pi_tf = nn(Q1, [hidden] * self.layers + [1], reuse=tf.AUTO_REUSE)
                Q2 = tf.concat(axis=1, values=Q_tfs)
                self.Q_tf   = nn(Q2, [hidden] * self.layers + [1], reuse=tf.AUTO_REUSE)

            elif conn_type == 'fc2':
                Q1 = tf.concat(axis=1, values=Q_pi_tfs)
                self.Q_pi_tf = nn(Q1, [hidden] * self.layers + [1], reuse=tf.AUTO_REUSE)
                Q2 = tf.concat(axis=1, values=Q_tfs)
                self.Q_tf   = nn(Q2, [hidden] * self.layers + [1], reuse=tf.AUTO_REUSE)

            elif conn_type=='layered':
                Q_i_1 = Q_pi_tfs[0]
                Q_i_2 = Q_tfs[0]

                for i in range(1, self.n_arms):
                    Q_1 = tf.concat(axis=1, values=[Q_i_1, Q_pi_tfs[i]])
                    Q_i_1 = nn(Q_1, [hidden] * self.layers + [1], reuse=tf.AUTO_REUSE)

                    Q_2 = tf.concat(axis=1, values=[Q_i_2, Q_tfs[i]])
                    Q_i_2 = nn(Q_2, [hidden] * self.layers + [1], reuse=tf.AUTO_REUSE)

                self.Q_pi_tf = Q_i_1
                self.Q_tf    = Q_i_2

        total_params()

class ActorCriticArea:
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

        # Raw observations
        #o_raw = self.o_tf[...,:self.env.o_ndx]
        # Joint poses
        joint_poses = self.o_tf[...,self.env.o_ndx:]
        joint_poses = narm_reshape(joint_poses, n_arms)

        # Normalization
        o = self.o_stats.normalize(self.o_tf)[...,:self.env.o_ndx]
        g = self.g_stats.normalize(self.g_tf)

        # Expose for debugging
        self.o_normalized = o
        self.g_normalized = g

        # Reshape inputs for the number of arms
        u = narm_reshape(self.u_tf, n_arms)
        o = narm_reshape(o, n_arms)
        g = narm_reshape(g, n_arms+1)
        
        # Outputs
        pi_tfs    = [None] * (n_arms-1)
        Q_pi_tfs  = [None] * (n_arms-1)
        Q_tfs     = [None] * (n_arms-1)
        
        for i in range(n_arms-1):

            if i != (n_arms-2):
                o_i = o[:, i]
                u_i = u[:, i]
                u_len = u_i.shape.as_list()[1]
            else:
                o_i = tf.layers.Flatten()(o[:, i:])
                u_i = tf.layers.Flatten()(u[:, i:])
                u_len = u_i.shape.as_list()[1]
            
            g_i = g[:, i]
            import pdb; pdb.set_trace()

            input_pis_i = tf.concat(axis=1, values=[o_i, g_i])
            
            hidden = solve_quadratic(self.hidden, self.layers, dimo, dimg, dimu, n_arms, u_len)
            with tf.variable_scope('pi{}'.format(i)):
                pi_tfs[i] = self.max_u * tf.tanh(nn(
                    input_pis_i, [hidden] * self.layers + [u_len]))

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
            self.Q_pi_tf = sum(Q_pi_tfs)
            self.Q_tf    = sum(Q_tfs)

        total_params()



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

        # Raw observations
        #o_raw = self.o_tf[...,:self.env.o_ndx]
        # Joint poses
        joint_poses = self.o_tf[...,self.env.o_ndx:]
        joint_poses = narm_reshape(joint_poses, n_arms)

        # Normalization
        o = self.o_tf[...,:self.env.o_ndx] #self.o_stats.normalize(self.o_tf)[...,:self.env.o_ndx]
        g = self.g_tf #self.g_stats.normalize(self.g_tf)

        # Expose for debugging
        self.o_normalized = o
        self.g_normalized = g

        # Reshape inputs for the number of arms
        u = narm_reshape(self.u_tf, n_arms)
        o = narm_reshape(o, n_arms)
        #g = narm_reshape(g, n_arms+1)
        
        # Compute goal in end-effector frame
        l  = self.o_tf[:,self.env.o_ndx:]
        end_eff = compute_end_eff_pos(l, o, g, self.n_arms)
        
        # Outputs
        pi_tfs    = [None] * n_arms
        Q_pi_tfs  = [None] * n_arms
        Q_tfs     = [None] * n_arms
        
        # Gradient of the end-effector with respect to states
        #L = tf.norm(g - end_eff, axis=1, keepdims=True)
        L = tf.reduce_sum(tf.square(g-end_eff), axis=1, keepdims=True)
        grads = tf.gradients(L, o)[0]
        
        #grads_x = tf.gradients(end_eff[:,0], o)[0]
        #grads_y = tf.gradients(end_eff[:,1], o)[0]
        #grads  = tf.concat(axis=2, values=[grads_x, grads_y])
        
        for i in range(n_arms):
            o_i = o[:, i]
            u_i = u[:, i]
            
            # Differentiation chain for method
            g_i = grads[:, i]

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
