import os
import mujoco_py
from gym import utils
from gym.envs.robotics import utils as gerutils
from gym.envs.robotics import fetch_env
from gym.envs.robotics.robot_env import fullpath_from_rel
import numpy as np

def goal_distance_3d(goal_a, goal_b):
    #if not goal_a.shape == goal_b.shape:
    #    import pdb; pdb.set_trace()
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach-actuated.xml')


class FetchReachActEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', **kwargs):
        # Set env o_ndx
        self.o_ndx = None

        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        model = mujoco_py.load_model_from_path(fullpath_from_rel(MODEL_XML_PATH))
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            n_actions=len(model.actuator_names), **kwargs)
        utils.EzPickle.__init__(self)


    def _set_action(self, action):
        assert action.shape == self.action_space.shape
        action = action 
        gerutils.full_ctrl_set_action(self.sim, action)


    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance_3d(achieved_goal, desired_goal)
        return d #(d < self.distance_threshold).astype(np.float32)
        
    
    def _get_obs(self):
        obs = super(FetchReachActEnv, self)._get_obs()
        joint_transforms, joint_qpos, joint_jacp, end_pose  = gerutils.get_joint_xposes(self.sim)
        
        # Traditional observation
        observation = np.asarray(joint_qpos)
        observation = [[np.sin(o), np.cos(o)] for o in observation]
        observation = np.asarray(observation).flatten()
        jacp = np.asarray(joint_jacp).flatten()
        end_pose = np.asarray(end_pose)

        if self.o_ndx is None:
            self.o_ndx = np.prod(observation.shape)
            self.o_ndx2 = self.o_ndx + np.prod(jacp.shape)

        obs['observation'] = np.concatenate((observation, jacp, end_pose), 0)
        return obs 
