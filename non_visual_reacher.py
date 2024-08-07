


import mujoco
import mujoco.viewer
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import glfw
import argparse
WIDTH = 500
HEIGHT = 500

class FrankaPandaEnv(gym.Env):
    def __init__(self, render=False, manual_mode = False):
        super(FrankaPandaEnv, self).__init__()
        
        # Load the Franka Panda model from MuJoCo Menagerie
        model_path = "env.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32)
        self.view = self._render_env(not manual_mode) if render else None
        
    def reset(self):
        default_kf = self.model.keyframe("default")
        self.data.qpos = default_kf.qpos.copy()
        self.data.ctrl = default_kf.ctrl.copy()
        ## Change Target Position
        self.model.site_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')] = [ np.random.uniform(0.5, 0.8),
                                                                                                          np.random.uniform(0, 0.3),
                                                                                                          np.random.uniform(0.5, 0.8)]
        print(self.model.site_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')])
        mujoco.mj_forward(self.model, self.data)
        self._sync_view()
        return self._get_obs()
    
    def step(self, action):
        self.data.ctrl[:] = action * 0.2
        mujoco.mj_step(self.model, self.data, 20)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        self._sync_view()
        return obs, reward, done, info

    def _render_env(self, disable_panels=True):
        return mujoco.viewer.launch_passive(self.model, self.data, show_left_ui= not disable_panels, show_right_ui=not disable_panels)
    def _sync_view(self):
        if self.view and self.view.is_running():
            with self.view.lock():
                self.view.sync()
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _compute_reward(self):
        # Define a simple reward function
        robotic_arm_pointer = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')]
        target = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')]

        # Calculate the distance
        distance = np.linalg.norm(target - robotic_arm_pointer)
        return -1 * distance * 10
    
    def _is_done(self):
        robotic_arm_pointer = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')]
        target = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')]
        distance = np.linalg.norm(target - robotic_arm_pointer)
        print(robotic_arm_pointer)
        return distance < 0.05 or np.any(robotic_arm_pointer > 1)
    

def test_env():
    env = FrankaPandaEnv(render=True)
    obs = env.reset()
    step = 0
    while True:
        # pass
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step += 1
        if done:
            print(obs, reward, done)
            _ = env.reset()
            time.sleep(0.2)
        
def run_manual():
    env = FrankaPandaEnv(render=True, manual_mode=True)
    obs = env.reset()
    while env.view.is_running():
        mujoco.mj_step(env.model, env.data, 20)
        print(f"Reward: {env._compute_reward()}, Done: {env._is_done()}")
        env._sync_view()

if __name__ == "__main__":
# Create an instance of the environment and test it
    
    parser = argparse.ArgumentParser(description="Update the position of a site in MuJoCo simulation.", argument_default=False)
    parser.add_argument('--manual', type=bool, required=False, help="Specify rendering in manual mode or not")
    args = parser.parse_args()

    if args.manual == True:
        run_manual()
    else:
        test_env()
            
    



