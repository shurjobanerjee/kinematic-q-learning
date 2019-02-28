import numpy as np
from gym import error
from pprint import pprint

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))




def apply_trans(trans, x):
    rot, pos = trans
    return rot.dot(x) + pos


def invert_trans(trans):
    rot, pos = trans
    return rot.T, -rot.T.dot(pos)


def apply_inv_trans(trans, x):
    rot, pos = trans
    return x - pos
    #return rot.T.dot(x) + pos
    #return rot.T.dot(x - pos)


class Trans:
    __slots__ = ("trans",)
    def __init__(self, rot, pos):
        self.trans = (rot, pos)

    def __call__(self, x):
        return apply_trans(self.trans, x)

    def invert(self):
        return type(self)(*invert_trans(self.trans))

    def apply_inv(self, x):
        return apply_inv_trans(self.trans,  x)

class PosT:
    __slots__ = ("trans",)
    def __init__(self, rot, pos):
        self.trans = pos

    def __call__(self, x):
        return x + self.trans

    def apply_inv(self, x):
        return x - self.trans


def rotmat(theta_y):
    return np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]])


def get_jacp(sim):
    return sim.data.get_site_jacp('robot0:grip').copy().reshape((3,-1)).T

def get_joint_xposes(sim, jacp=None):
    
    joint_jacobians_all = jacp if jacp is not None else get_jacp(sim) 
    
    joint_qpos, joint_qvel, joint_jacps = [], [], []
    for i in range(len(sim.model.actuator_names)):
        jidx = sim.model.actuator_trnid[i, 0]
        qposidx = sim.model.jnt_qposadr[jidx]
        jacobian = joint_jacobians_all[qposidx]

        joint_qpos.append(sim.data.qpos[qposidx])
        joint_qvel.append(sim.data.qvel[qposidx])
        joint_jacps.append(jacobian)
    
    return [np.asarray(x) for x in [joint_qpos, joint_qvel, joint_jacps]]

def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)



def full_ctrl_set_action(sim, action):
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    full_ctrl_set_action(sim, action)


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)
        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]
        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
