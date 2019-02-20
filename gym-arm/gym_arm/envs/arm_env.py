"""
Environment for Robot Arm.
You can customize this script in a way you want.

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/


Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np
import pyglet
import gym
from gym import error, spaces
from gym.utils import seeding

#FIXME
from params import Params

pyglet.clock.set_fps_limit(15)

def box(obs):
    return spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

def goal_distance_2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class ArmEnv(gym.GoalEnv):
    action_bound = [-1, 1]
    action_dim = 2
    dt = 1 # refresh rate
    arml  = 1
    viewer = None

    def __init__(self, 
                 reward_type='sparse', 
                 distance_threshold=50, 
                 n_arms=Params.n_arms, 
                 visible=True,  
                 wrapper_kwargs={}, 
                 **kwargs):

        self.n_arms = n_arms
        self.arm_info = np.zeros((n_arms, 4))
        self.arm_i = self.arml // n_arms
        self.arm_info[:, 0] = self.arm_i
        self.relative = Params.parts #FIXME
        
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
        achieved_goal = obs['achieved_goal'].copy()
        desired_goal  = obs['desired_goal'].copy()

        info = dict(is_success=goal_distance_2d(achieved_goal, desired_goal))
        reward = self.compute_reward(achieved_goal, desired_goal, info)

        self.last_info = info

        return obs, reward, done, info
    
    def random_point(self):
        """Random point selection on unit disk."""
        r = np.random.random() 
        t = np.random.random() * 2*np.pi 
        sr = np.sqrt(r) # Unbiased disk point sampling
        p = np.asarray([sr*self.arml*np.cos(t), sr*self.arml*np.sin(t)]) 
        return p

    def random_kinematics(self):
        # Random points that are achievable via the current arm configs
        self.arm_info[:, 1] = 2 * np.pi * np.random.random(self.n_arms)
        self.forward_kinematics()
    
    def forward_kinematics(self):
        armrad = 0
        center_coord = 0
        for i in range(self.n_arms):
            armrad +=  self.arm_info[i, 1]
            armdx_dy = np.array([self.arm_info[i, 0] * np.cos(armrad),
                                 self.arm_info[i, 0] * np.sin(armrad)])
            self.arm_info[i, 2:4] = center_coord + armdx_dy
            center_coord = self.arm_info[i, 2:4]
        
    def reset(self):
        # Desired Goal
        self.point_info[:] = self.random_point()
        # Starting orientation
        self.random_kinematics()
        return self._get_obs()

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, 
                                 self.point_info, 
                                 self.n_arms, 
                                 self.visible)

        self.viewer.set_info(self.last_info)
        self.viewer.render()
        
        if mode == 'rgb_array':
            return self.viewer.rgb_array()

    #def sample_action(self):
    #    return np.random.uniform(*self.action_bound, size=self.action_dim)

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_obs(self):
        
        # Observations
        observation = self.arm_info[:,1].copy()
        # Global coordinates
        achieved_goal = self.arm_info[-1][2:4].copy()
        desired_goal  = self.point_info.copy()
        
        return dict(
                achieved_goal = achieved_goal,
                desired_goal = desired_goal,
                observation = observation, 
            )



class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5
    point_l = 15
    viewer_xy = (600, 600)

    def __init__(self, arm_info, point_info, n_arms, visible):
        width, height = self.viewer_xy
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm', vsync=False, visible=visible)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.point_info = point_info

        self.n_arms = n_arms

        self.center_coord = np.array((min(width, height)/2, ) * 2)
        self.batch = pyglet.graphics.Batch()
        
        point_box = [0]*8
        arm_box = [None]*self.n_arms
        for i in range(n_arms):
            arm_box[i] = [0]*8

        c1, c2, c3 = (249, 0, 255)*4, (86, 109, 249)*4, (249, 39, 65)*4
        
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point_box), ('c3B', c2))
        
        self.arm = [None]*self.n_arms
        for i in range(n_arms):
            self.arm[i] = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm_box[i]), ('c3B', c1))
        self.info = None

    def set_info(self, info):
        self.info = info

    def set_arm_info(self, arm_info):
        self.arm_info = arm_info

    def render(self):
        pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.fps_display.draw()
        self.label.draw()

    def rgb_array(self):
        return pyglet.image.get_buffer_manager().get_color_buffer()

    def _update_arm(self):
        # Draw goal
        point_l = self.point_l
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)
        self.point.vertices = point_box
        
        # Draw arm boxes
        arm_coord = [None]*self.n_arms
        arm_box   = [None]*self.n_arms
        arm_thick_rad = [None]*self.n_arms
        center_coord = self.center_coord

        # Local to global angles
        global_angles = np.cumsum(self.arm_info[:, 1])
        
        for i in range(self.n_arms):
            arm_coord[i] = (*center_coord, *(300*self.arm_info[i, 2:4]))
            arm_thick_rad[i] = np.pi / 2 - global_angles[i]
        
            x01 = arm_coord[i][0] - np.cos(arm_thick_rad[i]) * self.bar_thc
            y01 = arm_coord[i][1] + np.sin(arm_thick_rad[i]) * self.bar_thc

            x02 = arm_coord[i][0] + np.cos(arm_thick_rad[i]) * self.bar_thc
            y02 = arm_coord[i][1] - np.sin(arm_thick_rad[i]) * self.bar_thc

            x11 = arm_coord[i][2] + np.cos(arm_thick_rad[i]) * self.bar_thc
            y11 = arm_coord[i][3] - np.sin(arm_thick_rad[i]) * self.bar_thc

            x12 = arm_coord[i][2] - np.cos(arm_thick_rad[i]) * self.bar_thc
            y12 = arm_coord[i][3] + np.sin(arm_thick_rad[i]) * self.bar_thc
        
            arm_box[i] = (x01, y01, x02, y02, x11, y11, x12, y12)

            self.arm[i].vertices = arm_box[i]

            center_coord = self.arm_info[i, 2:4] * 300

        #Draw info
        self.label = pyglet.text.Label("{:2f}".format(self.info['is_success']),
                                  font_name='Times New Roman',
                                  font_size=36,
                                  x=self.width//2, y=self.height//2,
                                  anchor_x='center', anchor_y='center',
                                  color=(0, 0, 0, 255))


