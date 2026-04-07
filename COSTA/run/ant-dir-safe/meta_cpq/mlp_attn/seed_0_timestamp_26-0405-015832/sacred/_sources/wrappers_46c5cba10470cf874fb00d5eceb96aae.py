import numpy as np
#import gym
#from gym import Env
#from gym.spaces import Box

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box

# import mujoco_py
import mujoco
from rlkit.core.serializable import Serializable


class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        # self._wrapped_env = wrapped_env
        # self.action_space = self._wrapped_env.action_space
        # self.observation_space = self._wrapped_env.observation_space

        # Use getattr with a fallback to avoid the AttributeError
        self.action_space = getattr(self._wrapped_env, 'action_space', None)
        self.observation_space = getattr(self._wrapped_env, 'observation_space', None)
        if self.observation_space is None:
            print("WARNING: observation_space not found. Forcing initialization...")
            # Some old environments use these methods to build the spaces
            if hasattr(self._wrapped_env, '_set_observation_space'):
                self._wrapped_env._set_observation_space()
                self.observation_space = self._wrapped_env.observation_space
                
        self._max_episode_steps = self._wrapped_env._max_episode_steps

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, 'log_diagnostics'):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()


# class NormalizedBoxEnv(ProxyEnv, Serializable):
#     """
#     Normalize action to in [-1, 1].
#
#     Optionally normalize observations and scale reward.
#     """
#     def __init__(
#             self,
#             env,
#             reward_scale=1.,
#             obs_mean=None,
#             obs_std=None,
#     ):
#         # self._wrapped_env needs to be called first because
#         # Serializable.quick_init calls getattr, on this class. And the
#         # implementation of getattr (see below) calls self._wrapped_env.
#         # Without setting this first, the call to self._wrapped_env would call
#         # getattr again (since it's not set yet) and therefore loop forever.
#         self._wrapped_env = env
#         # Or else serialization gets delegated to the wrapped_env. Serialize
#         # this env separately from the wrapped_env.
#         self._serializable_initialized = False
#         Serializable.quick_init(self, locals())
#         ProxyEnv.__init__(self, env)
#         self._should_normalize = not (obs_mean is None and obs_std is None)
#         if self._should_normalize:
#             if obs_mean is None:
#                 obs_mean = np.zeros_like(env.observation_space.low)
#             else:
#                 obs_mean = np.array(obs_mean)
#             if obs_std is None:
#                 obs_std = np.ones_like(env.observation_space.low)
#             else:
#                 obs_std = np.array(obs_std)
#         self._reward_scale = reward_scale
#         self._obs_mean = obs_mean
#         self._obs_std = obs_std
#         ub = np.ones(self._wrapped_env.action_space.shape)
#         self.action_space = Box(-1 * ub, ub)
#
#     def estimate_obs_stats(self, obs_batch, override_values=False):
#         if self._obs_mean is not None and not override_values:
#             raise Exception("Observation mean and std already set. To "
#                             "override, set override_values to True.")
#         self._obs_mean = np.mean(obs_batch, axis=0)
#         self._obs_std = np.std(obs_batch, axis=0)
#
#     def _apply_normalize_obs(self, obs):
#         return (obs - self._obs_mean) / (self._obs_std + 1e-8)
#
#     def __getstate__(self):
#         d = Serializable.__getstate__(self)
#         # Add these explicitly in case they were modified
#         d["_obs_mean"] = self._obs_mean
#         d["_obs_std"] = self._obs_std
#         d["_reward_scale"] = self._reward_scale
#         return d
#
#     def __setstate__(self, d):
#         Serializable.__setstate__(self, d)
#         self._obs_mean = d["_obs_mean"]
#         self._obs_std = d["_obs_std"]
#         self._reward_scale = d["_reward_scale"]
#
#     def step(self, action):
#         lb = self._wrapped_env.action_space.low
#         ub = self._wrapped_env.action_space.high
#         scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
#         scaled_action = np.clip(scaled_action, lb, ub)
#
#         wrapped_step = self._wrapped_env.step(scaled_action)
#         next_obs, reward, done, info = wrapped_step
#         if self._should_normalize:
#             next_obs = self._apply_normalize_obs(next_obs)
#         return next_obs, reward * self._reward_scale, done, info
#
#     def __str__(self):
#         return "Normalized: %s" % self._wrapped_env
#
#     def log_diagnostics(self, paths, **kwargs):
#         if hasattr(self._wrapped_env, "log_diagnostics"):
#             return self._wrapped_env.log_diagnostics(paths, **kwargs)
#         else:
#             return None
#
#     def __getattr__(self, attrname):
#         return getattr(self._wrapped_env, attrname)

class NormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_absmax=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_absmax is None)
        if self._should_normalize:
            if obs_absmax is None:
                obs_absmax = np.zeros_like(env.observation_space.high)
            else:
                obs_mean = np.array(obs_absmax)
        self._reward_scale = reward_scale
        self._obs_absmax = obs_absmax
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_absmax is not None and not override_values:
            raise Exception("Observation absmax already set. To "
                            "override, set override_values to True.")
        self._obs_absmax = np.max(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return obs / (self._obs_absmax + 1e-8)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_absmax"] = self._obs_absmax
        d["_reward_scale"] = self._reward_scale
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_absmax = d["_obs_absmax"]
        self._reward_scale = d["_reward_scale"]

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

class CameraWrapper(object):

    def __init__(self, env,  *args, **kwargs):
        self._wrapped_env = env
        self.initialize_camera()

    # def get_image(self, width=256, height=256, camera_name=None):
    #     # use sim.render to avoid MJViewer which doesn't seem to work without display
    #     return self.sim.render(
    #         width=width,
    #         height=height,
    #         camera_name=camera_name,
    #     )
    
    # import mujoco

    def get_image(self, width=256, height=256, camera_name=None):
        # 1. Initialize the renderer if it hasn't been created yet
        # We store it as an attribute so we don't recreate it every frame
        if not hasattr(self, 'renderer'):
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        # 2. Update the visual scene with the current simulation state
        # This replaces the internal 'sim.render' logic
        self.renderer.update_scene(self.data, camera=camera_name)
        
        # 3. Render and return the pixel array (RGB)
        return self.renderer.render()

    # def initialize_camera(self):
    #     # set camera parameters for viewing
    #     sim = self.sim
    #     viewer = mujoco_py.MjRenderContextOffscreen(sim)
    #     camera = viewer.cam
    #     camera.type = 1
    #     camera.trackbodyid = 0
    #     camera.elevation = -20
    #     sim.add_render_context(viewer)

    # import mujoco

    def initialize_camera(self):
        # In the new API, we create a Renderer object
        # We define the resolution (height, width) here
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # Access the camera parameters directly
        # 'mjvCamera' is the equivalent of 'viewer.cam'
        camera = self.renderer.scene.camera
        
        # Set camera parameters (similar to before)
        # camera.type: 0=free, 1=tracking, 2=fixed
        camera.type = 1 
        camera.trackbodyid = 0
        camera.elevation = -20
        
        # Note: There is no need for 'sim.add_render_context' anymore.
        # The renderer is now a standalone tool that acts on (model, data).


    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)
