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

from params import Params
#pyglet.clock.set_fps_limit(10000)
pyglet.clock.set_fps_limit(15)



def box(obs):
    return spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape

    return np.linalg.norm(goal_a[...,:2] - goal_b[...,:2], axis=-1)

class ArmEnv(gym.GoalEnv):
    action_bound = [-1, 1]
    action_dim = 2
    state_dim = 7
    dt = 1  # refresh rate
    arml  = 300
    viewer = None
    viewer_xy = (600, 600)
    get_point = False
    mouse_in = np.array([False])
    point_l = 15
    grab_counter = 0

    def __init__(self, mode='hard', reward_type='sparse', distance_threshold=50, n_arms=3, visible=True,  wrapper_kwargs={}, **kwargs):
        # node1 (l, d_rad, x, y),
        # node2 (l, d_rad, x, y)
        self.mode = mode
        n_arms = Params.n_arms #FIXME
        self.n_arms = n_arms
        self.arm_info = np.zeros((n_arms, 4))
        for i in range(n_arms):
            self.arm_info[i, 0] = self.arml // n_arms
        self.point_info = np.array([250, 303])
        self.point_info_init = self.point_info.copy()
        self.center_coord = np.array(self.viewer_xy)/2
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        self.relative = Params.parts
        self.visible = visible

        # Required for Goal-Env
        self.action_space = spaces.Box(-1., 1., shape=(n_arms,), dtype='float32')
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=box(obs['desired_goal']),
            achieved_goal=box(obs['achieved_goal']),
            observation=box(obs['observation']),
        ))

    def random_point(self):
        r = np.random.random() * self.arml #self.arm_info[:, 0].sum()
        t = np.random.random() * 2*np.pi
        p = np.asarray([r*np.cos(t), r*np.sin(t)]) + self.center_coord
        return p

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
    
    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
    
    def step(self, action):
        # action = (node1 angular v, node2 angular v)
        action = np.clip(action, *self.action_bound)
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2

        self.forward_kinematics()

        # Returns
        obs = self._get_obs()
        done = False
        info = dict(is_success=goal_distance(obs['achieved_goal'], obs['desired_goal']))
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

        self.last_info = info

        return obs, reward, done, info
    
    def forward_kinematics(self):
        armdx_dy = [None] * self.n_arms
        for i in range(self.n_arms):
            armrad = self.arm_info[i, 1]
            armdx_dy[i] = np.array([self.arm_info[i, 0] * np.cos(armrad),
                                 self.arm_info[i, 0] * np.sin(armrad)])
        
        center_coord = self.center_coord
        for i in range(self.n_arms):
            self.arm_info[i, 2:4] = center_coord + armdx_dy[i]
            center_coord = self.arm_info[i, 2:4]
        
    def reset(self):
        self.get_point = False
        self.grab_counter = 0
        self.mode = 'easy'
        if self.mode == 'hard':
            #pxy = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)
            self.point_info[:] = self.random_point() #pxy
        else:
            armrad = np.random.rand(self.n_arms) * np.pi * 2
            for i in range(self.n_arms):
                self.arm_info[i, 1] = armrad[i]
            self.forward_kinematics()
        
            self.point_info[:] = self.random_point() #pxy
            #self.point_info[:] = self.point_info_init
        return self._get_obs()

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_l, self.mouse_in, self.n_arms, self.visible)
        self.viewer.set_info(self.last_info)
        self.viewer.render()
        if mode == 'rgb_array':
            return self.viewer.rgb_array()

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_obs(self, relative=True):
        # return the distance (dx, dy) between arm finger point with blue point
        obs = {}
        arm_end = self.arm_info[-1, 2:4] # End-effector position
        obs['achieved_goal'] = arm_end
        obs['desired_goal'] = self.point_info

        observation = np.zeros(self.n_arms * 2)
        for i in range(0, self.n_arms):
            armrad = self.arm_info[i, 1]
            observation[2*i]   = np.sin(armrad)
            observation[2*i+1] = np.cos(armrad) 
        
        obs['observation'] = observation

        #if self.relative:
        #    for g in ['achieved_goal', 'desired_goal']:
        #        obs_g = [obs[g]]
        #        for i in range(self.n_arms-2):
        #            obs_g += [obs[g] - self.arm_info[i, 2:4]]
        #        obs[g] = np.concatenate(obs_g)
        return obs

        

class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, arm_info, point_info, point_l, mouse_in, n_arms, visible):
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm', vsync=False, visible=visible)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.point_info = point_info
        self.mouse_in = mouse_in
        self.point_l = point_l

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

        for i in range(self.n_arms):
            arm_coord[i] = (*center_coord, *(self.arm_info[i, 2:4]))
            arm_thick_rad[i] = np.pi / 2 - self.arm_info[i, 1]
        
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

            center_coord = self.arm_info[i, 2:4]

        #Draw info
        self.label = pyglet.text.Label("{:2f}".format(self.info['is_success']),
                                  font_name='Times New Roman',
                                  font_size=36,
                                  x=self.width//2, y=self.height//2,
                                  anchor_x='center', anchor_y='center',
                                  color=(0, 0, 0, 255))


    #def on_key_press(self, symbol, modifiers):
    #    if symbol == pyglet.window.key.UP:
    #        self.arm_info[0, 1] += .1
    #        print(self.arm_info[:, 2:4] - self.point_info)
    #    elif symbol == pyglet.window.key.DOWN:
    #        self.arm_info[0, 1] -= .1
    #        print(self.arm_info[:, 2:4] - self.point_info)
    #    elif symbol == pyglet.window.key.LEFT:
    #        self.arm_info[1, 1] += .1
    #        print(self.arm_info[:, 2:4] - self.point_info)
    #    elif symbol == pyglet.window.key.RIGHT:
    #        self.arm_info[1, 1] -= .1
    #        print(self.arm_info[:, 2:4] - self.point_info)
    #    elif symbol == pyglet.window.key.Q:
    #        pyglet.clock.set_fps_limit(1000)
    #    elif symbol == pyglet.window.key.A:
    #        pyglet.clock.set_fps_limit(30)

    #def on_mouse_motion(self, x, y, dx, dy):
    #    self.point_info[:] = [x, y]

    #def on_mouse_enter(self, x, y):
    #    self.mouse_in[0] = True

    #def on_mouse_leave(self, x, y):
    #    self.mouse_in[0] = False

