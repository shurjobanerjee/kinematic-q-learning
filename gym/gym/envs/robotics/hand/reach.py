import os
import numpy as np

from gym import utils
from gym.envs.robotics import hand_env
from gym.envs.robotics.utils import robot_get_obs
from gym.envs.robotics import utils as gerutils
from gym.envs.robotics.utils import ObsReshaper


FINGERTIP_SITE_NAMES = [
    'robot0:S_fftip',
    'robot0:S_mftip',
    'robot0:S_rftip',
    'robot0:S_lftip',
    'robot0:S_thtip',
]


DEFAULT_INITIAL_QPOS = {
    'robot0:WRJ1': -0.16514339750464327,
    'robot0:WRJ0': -0.31973286565062153,
    'robot0:FFJ3': 0.14340512546557435,
    'robot0:FFJ2': 0.32028208333591573,
    'robot0:FFJ1': 0.7126053607727917,
    'robot0:FFJ0': 0.6705281001412586,
    'robot0:MFJ3': 0.000246444303701037,
    'robot0:MFJ2': 0.3152655251085491,
    'robot0:MFJ1': 0.7659800313729842,
    'robot0:MFJ0': 0.7323156897425923,
    'robot0:RFJ3': 0.00038520700007378114,
    'robot0:RFJ2': 0.36743546201985233,
    'robot0:RFJ1': 0.7119514095008576,
    'robot0:RFJ0': 0.6699446327514138,
    'robot0:LFJ4': 0.0525442258033891,
    'robot0:LFJ3': -0.13615534724474673,
    'robot0:LFJ2': 0.39872030433433003,
    'robot0:LFJ1': 0.7415570009679252,
    'robot0:LFJ0': 0.704096378652974,
    'robot0:THJ4': 0.003673823825070126,
    'robot0:THJ3': 0.5506291436028695,
    'robot0:THJ2': -0.014515151997119306,
    'robot0:THJ1': -0.0015229223564485414,
    'robot0:THJ0': -0.7894883021600622,
}


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('hand', 'reach.xml')


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class HandReachEnv(hand_env.HandEnv, utils.EzPickle):
    def __init__(
        self, distance_threshold=0.001, n_substeps=20, relative_control=False,
        initial_qpos=DEFAULT_INITIAL_QPOS, reward_type='sparse', **kwargs
    ):
        self.reshaper = None
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        hand_env.HandEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=n_substeps, initial_qpos=initial_qpos,
            relative_control=relative_control, **kwargs)
        utils.EzPickle.__init__(self)

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()
    
    def _get_joint_jacp(self):
        jacps = [self.sim.data.get_site_jacp(name).reshape((3,-1)).T \
                for name in FINGERTIP_SITE_NAMES]
        jacps = np.concatenate(jacps, 1)
        return jacps

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        self.initial_goal = self._get_achieved_goal().copy()
        self.palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()
    
    def get_obs_dict(self):
        robot_jacp = self._get_joint_jacp()
        joint_qpos, joint_qvel, joint_jacp  = gerutils.get_joint_xposes(self.sim, robot_jacp)
        observation = np.asarray(list(zip(joint_qpos, joint_qvel)))
        end_eff = self._get_achieved_goal().ravel()
        return dict(observation = observation,
                    jacp        = joint_jacp,
                    end_eff     = end_eff)

    def _get_obs(self):
        """
        Observations wrt to actuator
        """
        obs_dict = self.get_obs_dict()
        if self.reshaper is None:
            self.reshaper = ObsReshaper(**obs_dict)
        observation = self.reshaper.linearize(**obs_dict) 
        achieved_goal = self._get_achieved_goal().ravel()

        # Add jacobian information
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    #def _get_obs(self):
    #    robot_qpos, robot_qvel = robot_get_obs(self.sim)
    #    achieved_goal = self._get_achieved_goal().ravel()
    #    return {
    #        'observation': observation.copy(),
    #        'achieved_goal': achieved_goal.copy(),
    #        'desired_goal': self.goal.copy(),
    #    }
    #    
    #    obs = super(FetchReachActEnv, self)._get_obs()
    #    joint_qpos, joint_qvel, joint_jacp, grip_pose  = gerutils.get_joint_xposes(self.sim)
    #    
    #    # Same observation as for hand tasks
    #    #observation = np.concatenate((joint_qpos, joint_qvel, grip_pose), 0)
    #    #observation = [[np.sin(o), np.cos(o)] for o in joint_qpos]
    #    observation = np.asarray([list(q) for q in zip(joint_qpos, joint_qvel)]).flatten()
    #    observation = np.asarray(observation).flatten()
    #    jacp = np.asarray(joint_jacp).flatten()
    #    if self.o_ndx is None:
    #        self.o_ndx = np.prod(observation.shape)
    #        self.o_ndx2 = self.o_ndx + np.prod(jacp.shape)

    #    obs['observation'] = np.concatenate((observation, jacp, grip_pose), 0)
    #    return obs 
    
    def _sample_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = self.np_random.choice(finger_names)

        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)

        # Slightly move meeting goal towards the respective finger to avoid that they
        # overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        for idx in [thumb_idx, finger_idx]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        if self.np_random.uniform() < 0.1:
            # With some probability, ask all fingers to move back to the origin.
            # This avoids that the thumb constantly stays near the goal position already.
            goal = self.initial_goal.copy()
        return goal.flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return d #(d < self.distance_threshold).astype(np.float32)

    def _render_callback(self):
        # Visualize targets.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        goal = self.goal.reshape(5, 3)
        for finger_idx in range(5):
            site_name = 'target{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = goal[finger_idx] - sites_offset[site_id]

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(5, 3)
        for finger_idx in range(5):
            site_name = 'finger{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]
        self.sim.forward()
