


import mujoco
import mujoco.viewer
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import glfw
WIDTH = 500
HEIGHT = 500

class FrankaPandaEnv(gym.Env):
    def __init__(self):
        super(FrankaPandaEnv, self).__init__()
        
        # Load the Franka Panda model from MuJoCo Menagerie
        model_path = "env.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32)
        
    def reset(self):
        default_kf = self.model.keyframe("default")
        print(default_kf.qpos)
        # mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = default_kf.qpos.copy()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
    
    def step(self, action):
        self.data.ctrl[:] = action * 0.4
        mujoco.mj_step(self.model, self.data, 20)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        return obs, reward, done, info

        
    
    def render_env(self):
        return mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=True)
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _compute_reward(self):
        # Define a simple reward function
        robotic_arm_pointer = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')]
        target = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')]

        # Calculate the distance
        distance = np.linalg.norm(target - robotic_arm_pointer)
        return -1 * distance
    
    def _is_done(self):
        robotic_arm_pointer = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')]
        target = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')]
        distance = np.linalg.norm(target - robotic_arm_pointer)
        return distance < 0.01
    




if __name__ == "__main__":
# Create an instance of the environment and test it
    env = FrankaPandaEnv()
    view = env.render_env()
    # glfw.set_window_size(view, WIDTH, HEIGHT) 
    obs = env.reset()
    step = 0
    while view.is_running():
        # pass
        # action = env.action_space.sample()
        mujoco.mj_step(env.model, env.data, 20)
        # obs, reward, done, info = env.step(action)
        # print(obs, reward, done)
        # reward = env._compute_reward()
        # print("Reward  Done:", reward, env._is_done())
        # step += 1
        # if done:
        #     _ = env.reset()
        # time.sleep(0.1)
        with view.lock():
            view.sync()
            
    



