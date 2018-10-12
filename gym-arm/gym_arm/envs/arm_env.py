"""
Environment for Robot Arm.
You can customize this script in a way you want.

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/


Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.robotics.utils import ObsReshaper



def box(obs):
    return spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

def goal_distance_2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def rcumsum(x):
    return np.cumsum(x[::-1])[::-1] 

class ArmEnv(gym.GoalEnv):
    dt = 1 # refresh rate
    arml  = 1 # Unit disk
    viewer = None

    def __init__(self, 
                 reward_type='sparse', 
                 distance_threshold=1./20., 
                 n_arms=2, 
                 visible=True,
                 achievable=False,
                 wrapper_kwargs={},
                 conn_type=None,
                 parts=None,
                 **kwargs):

        self.n_arms = n_arms
        self.arm_info = np.zeros((n_arms, 4))
        self.desired_info = self.arm_info.copy() if achievable==True else None
        self.achievable = achievable
        self.conn_type = conn_type
        self.parts = parts

        self.arm_i = self.arml / n_arms
        self.arm_info[:, 0] = self.arm_i
        
        # Consistent reshaper
        self.reshaper = None
        
        # Reset whhen initializing
        self.point_info = np.zeros(2)
        self.reset()
        
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        self.visible = visible
        
        # Required for Goal-Env
        self.action_space = spaces.Box(-1., 1., shape=(n_arms,), dtype='float32')
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=box(obs['desired_goal']),
            achieved_goal=box(obs['achieved_goal']),
            observation=box(obs['observation']),
        ))

        # Observation indices required by model


    def get_obs_dict(self):
        """Couples together observations"""
        # Actual observation
        angles = self.arm_info[:,1:2].copy()
        # Additional information
        jacp = self.jacp()
        # End effector 
        end_eff=self.arm_info[-1,2:4].copy()
        
        return dict(observation=angles, 
                    jacp=jacp, 
                    end_eff=end_eff)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance_2d(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
    
    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance_2d(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
    
    def step(self, action,):
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2
        self.forward_kinematics()

        # Returns
        obs = self._get_obs()
        done = False

        # Get achieved goal and desired goal in terms of global coordinates
        achieved_goal = obs['achieved_goal'][:2].copy()
        desired_goal  = obs['desired_goal'][:2].copy()

        #info = dict(is_success=self._is_success(achieved_goal, desired_goal))
        info = dict(is_success=goal_distance_2d(achieved_goal, desired_goal))
        reward = self.compute_reward(achieved_goal, desired_goal, info)

        self.last_info = info

        return obs, reward, done, info
    
    def random_point(self):
        """Random point selection on unit disk."""
        if not self.achievable:
            r = np.random.random() 
            t = np.random.random() * 2*np.pi 
            sr = np.sqrt(r) # Unbiased disk point sampling
            p = np.asarray([sr*self.arml*np.cos(t), sr*self.arml*np.sin(t)]) 
        else:
            self.random_kinematics()
            self.desired_info = self.arm_info.copy()
            p = self.desired_info[-1, 2:4]

        return p

    def random_kinematics(self):
        # Random points that are achievable via the current arm configs
        self.arm_info[:, 1] = 2 * np.pi * np.random.random(self.n_arms)
        self.forward_kinematics()
    
    def forward_kinematics(self):
        ls     = self.arm_info[:, 0]
        thetas = self.arm_info[:, 1]
        
        # Functional form of forward kinematics
        self.arm_info[:, 2:4] = \
                np.asarray([np.cumsum(ls*np.cos(np.cumsum(thetas))), 
                            np.cumsum(ls*np.sin(np.cumsum(thetas)))]).T

        # armrad = 0
        # center_coord = 0
        # for i in range(self.n_arms):
        #     armrad +=  self.arm_info[i, 1]
        #     armdx_dy = np.array([self.arm_info[i, 0] * np.cos(armrad),
        #                          self.arm_info[i, 0] * np.sin(armrad)])
        #     self.arm_info[i, 2:4] = center_coord + armdx_dy
        #     center_coord = self.arm_info[i, 2:4]
    
    def jacp(self):
        """
        Analytical form of derivative of end-effector wrt angles
        """
        ls     = self.arm_info[:, 0]
        thetas = self.arm_info[:, 1]
        return np.asarray([-rcumsum(ls*np.sin(np.cumsum(thetas))), 
                            rcumsum(ls*np.cos(np.cumsum(thetas)))]).T
    
    def reset(self):
        # Desired Goal
        self.point_info[:] = self.random_point()
        # Starting orientation
        self.random_kinematics()
        return self._get_obs()

    def render(self, mode='human'):
        if self.viewer is None:
            from .viewer import Viewer
            self.viewer = Viewer(self.arm_info, 
                                 self.point_info, 
                                 self.n_arms, 
                                 self.visible,
                                 self.desired_info)

        self.viewer.set_vals(self.arm_info, self.desired_info, self.last_info)
        self.viewer.render()
        
        if mode == 'rgb_array':
            return self.viewer.rgb_array()

    #def sample_action(self):
    #    return np.random.uniform(*self.action_bound, size=self.action_dim)

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_obs(self):
        # Observations (includes jacobian of end-effector)
        observation = self.get_obs_dict()
        # Linearize observation
        if self.reshaper is None:
            self.reshaper = ObsReshaper(**observation)
        observation = self.reshaper.linearize(**observation)
        
        # Achieved and desired goals
        achieved_goal = self.arm_info[-1][2:4].copy()
        desired_goal  = self.point_info.copy()
        
        return dict(
                achieved_goal = achieved_goal,
                desired_goal = desired_goal,
                observation = observation, 
            )





