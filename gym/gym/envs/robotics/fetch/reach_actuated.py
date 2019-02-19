import os
from gym import utils
from gym.envs.robotics import utils as gerutils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach-actuated.xml')


class FetchReachActEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, n_actions=6)
        utils.EzPickle.__init__(self)

    def _set_action(self, action):
        assert action.shape == self.action_space.shape
        action = action 
        gerutils.full_ctrl_set_action(self.sim, action)
