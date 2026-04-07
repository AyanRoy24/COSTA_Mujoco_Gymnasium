import os
from os import path

# import mujoco_py
import mujoco
import numpy as np
#from gym.envs.mujoco import mujoco_env
from gymnasium.envs.mujoco import mujoco_env

from rlkit.core.serializable import Serializable

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


class MujocoEnv(mujoco_env.MujocoEnv, Serializable):
    """
    My own wrapper around MujocoEnv.

    The caller needs to declare
    """
    # ADD THIS BLOCK: This satisfies the assertion check in the base class
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
            self,
            model_path,
            frame_skip=1,
            model_path_is_local=True,
            automatically_set_obs_and_action_space=False,
            render_mode=None,
    ):
        if model_path_is_local:
            model_path = get_asset_xml(model_path)

        # --- THE FIX ---
        # If your error says "Expected 20", you can hardcode 20 here, 
        # or calculate it if you know your model's timestep:
        # self.metadata["render_fps"] = int(np.round(1.0 / (frame_skip * 0.002)))
        self.metadata["render_fps"] = 20
        
        if automatically_set_obs_and_action_space:
            mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip,observation_space=None, render_mode=render_mode) #, self.observation_space, self.action_space)
        else:
            """
            Code below is copy/pasted from MujocoEnv's __init__ function.
            """
            if model_path.startswith("/"):
                fullpath = model_path
            else:
                fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
            self.frame_skip = frame_skip
            # self.model = mujoco_py.MjModel(fullpath)
            self.model = mujoco.MjModel.from_xml_path(fullpath)
            # self.data = self.model.data
            self.data = mujoco.MjData(self.model)
            self.viewer = None

            self.metadata = {
                'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': int(np.round(1.0 / self.dt))
            }

            self.init_qpos = self.model.data.qpos.ravel().copy()
            self.init_qvel = self.model.data.qvel.ravel().copy()
            self._seed()

    def init_serialization(self, locals):
        Serializable.quick_init(self, locals)

    def log_diagnostics(self, *args, **kwargs):
        pass


def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)
