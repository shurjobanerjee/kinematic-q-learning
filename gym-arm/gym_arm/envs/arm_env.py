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
from gym.envs.robotics.utils import ObsReshaper


pyglet.clock.set_fps_limit(30)

def box(obs):
    return spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

def goal_distance_2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class ArmEnv(gym.GoalEnv):
    dt = 1 # refresh rate
    arml  = 1 # Unit disk
    viewer = None

    def __init__(self, 
                 reward_type='sparse', 
                 distance_threshold=1/20, 
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


    def get_shaped_observations(self):
        """Couples together observations"""
        # Actual observation
        angles = self.arm_info[:,1].copy()
        # Additional information
        joint_lengths = self.arm_info[:, 0]
        # End effector 
        return dict(observation=angles, 
                    joint_lengths=joint_lengths, 
                    end_eff=self.arm_info[-1,2:4].copy())

    def preprocess_observation_ndxs(self):
        obs, jp = self.get_shaped_observations()
        return np.prod(obs.shape), jp.shape


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

        info = dict(is_success=self._is_success(achieved_goal, desired_goal))
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
        # Observations
        observation = self.get_shaped_observations()
        # Linearize observation
        if self.reshaper is None:
            self.reshaper = ObsReshaper(**observation)
        observation = self.reshaper.linearize(**observation)

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
    scale_fac = 200
    bar_thc = 5
    point_l = 1/20
    viewer_xy = (2*scale_fac, 2*scale_fac)

    def __init__(self, arm_info, point_info, n_arms, visible, desired_info):
        width, height = self.viewer_xy
        super(Viewer, self).__init__(width, 
                                     height, 
                                     resizable=False, 
                                     caption='Arm', 
                                     vsync=False, 
                                     visible=visible)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.desired_info = desired_info
        self.point_info = point_info

        self.n_arms = n_arms
        
        self.center_coord = np.array((self.scale_fac, self.scale_fac))
        self.batch = pyglet.graphics.Batch()
        
        # Colors
        c1, c2, c3 = (249, 0, 255)*4, (86, 109, 249)*4, (249, 39, 65)*4
        
        # Objects
        point_box = [0]*8
        self.point = self.batch_add(point_box, c2)
        arm_box = [[0]*8 for i in range(self.n_arms)]
        self.arm = [self.batch_add(arm_box[i], c1) for i in range(self.n_arms)]
        
        if self.desired_info is not None:
            arm_box_des = [[0]*8 for i in range(self.n_arms)]
            self.arm_des = [self.batch_add(arm_box_des[i], c3) for i in range(self.n_arms)]

        self.info = None

    def batch_add(self, v2f, c3b):
        return self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', v2f), ('c3B', c3b))

    def set_vals(self, arm_info, desired_info, info):
        self.arm_info = arm_info.copy()
        self.desired_info = desired_info if desired_info is None else desired_info.copy()
        self.info = info.copy()

    def set_arm_info(self, arm_info):
        self.arm_info = arm_info

    def render(self):
        pyglet.clock.tick()
        self._update_sim()
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

    def scale(self, pts):
        return pts * self.scale_fac + self.scale_fac

    def _update_sim(self):
        self._make_goal(self.point_info, self.point_l)
        if self.desired_info is not None:
            self._update_arm(self.desired_info, self.arm_des)
        self._update_arm(self.arm_info, self.arm)


    def _make_goal(self, point_info, point_l):
        # Draw goal
        point_box = np.asarray([
                     point_info[0] - point_l, point_info[1] - point_l,
                     point_info[0] + point_l, point_info[1] - point_l,
                     point_info[0] + point_l, point_info[1] + point_l,
                     point_info[0] - point_l, point_info[1] + point_l])
        self.point.vertices = self.scale(point_box)

    def _update_arm(self, arm_info, arm):
        # Local to global angles
        global_angles = np.cumsum(arm_info[:, 1])
        arm_info[:,2:4] = self.scale(arm_info[:,2:4])

        # Draw arm boxes
        center_coord = self.center_coord
        for i in range(self.n_arms):
            arm_coord = (*(center_coord), *(arm_info[i, 2:4]))
            arm_thick_rad = np.pi / 2 - global_angles[i]
        
            x01 = arm_coord[0] - np.cos(arm_thick_rad) * self.bar_thc
            y01 = arm_coord[1] + np.sin(arm_thick_rad) * self.bar_thc

            x02 = arm_coord[0] + np.cos(arm_thick_rad) * self.bar_thc
            y02 = arm_coord[1] - np.sin(arm_thick_rad) * self.bar_thc

            x11 = arm_coord[2] + np.cos(arm_thick_rad) * self.bar_thc
            y11 = arm_coord[3] - np.sin(arm_thick_rad) * self.bar_thc

            x12 = arm_coord[2] - np.cos(arm_thick_rad) * self.bar_thc
            y12 = arm_coord[3] + np.sin(arm_thick_rad) * self.bar_thc
        
            arm_box = (x01, y01, x02, y02, x11, y11, x12, y12)

            arm[i].vertices = arm_box

            center_coord = arm_info[i, 2:4] 

        #Distance to goal
        self.label = pyglet.text.Label("{:2f}".format(self.info['is_success']),
                                  font_name='Times New Roman',
                                  font_size=36,
                                  x=self.width//2, y=self.height//2,
                                  anchor_x='center', anchor_y='center',
                                  color=(0, 0, 0, 255))


