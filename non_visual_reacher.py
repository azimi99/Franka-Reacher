


import mujoco
import mujoco.viewer
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import glfw
import argparse

ARM_VEL_LIMITS = np.array([2.61799, 2.61799, 2.61799, 2.61799, 3.14159, 3.14159, 3.14159, 0])
class FrankaPandaEnv(gym.Env):
    def __init__(self, render=False, manual_mode = False, frame_skip=20):
        super(FrankaPandaEnv, self).__init__()
        
        # Load the Franka Panda model from MuJoCo Menagerie
        model_path = "env.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._frame_skip = frame_skip
        self._dt = self.model.opt.timestep * self._frame_skip
        # Define the action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 3 + self.model.nq + self.model.nv,), dtype=np.float32)
        self._last_qpos = self.data.qpos.copy()
        self._arm_ctrlrange_min = self.model.actuator_ctrlrange[:, 0]
        self._arm_ctrlrange_max = self.model.actuator_ctrlrange[:, 1] 
        self.view = self._render_env(not manual_mode) if render else None
        
    def reset(self):
        default_kf = self.model.keyframe("default")
        self.data.qpos = default_kf.qpos.copy()
        self.data.ctrl = default_kf.ctrl.copy()
        ## Change Target Position
        self.model.site_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')] = [ np.random.uniform(0.7, 0.8),
                                                                                                          np.random.uniform(-0.2, 0.2),
                                                                                                          np.random.uniform(0.2, 0.6)]
        mujoco.mj_forward(self.model, self.data)
        self._sync_view()
        return self._get_obs()
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0) 
        # print(action.shape)
        action = (((action + 1) * 0.5) * ARM_VEL_LIMITS * 2) - ARM_VEL_LIMITS
        action = action * self._dt
        action *= 0.4 #np.clip(self._compute_distance(), 0, 0.4) # Scale down the action
        
        new_qpos = self._last_qpos[:8] + action
        clipped_new_qpos = np.clip(new_qpos, self._arm_ctrlrange_min, self._arm_ctrlrange_max)
        
        self.data.ctrl[:] = clipped_new_qpos
        mujoco.mj_step(self.model, self.data, self._frame_skip)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        done = self._is_done()
        info = {}
        self._sync_view()
        self._last_qpos = self.data.qpos.copy()
        return obs, reward, done, info

    def _render_env(self, disable_panels=True):
        return mujoco.viewer.launch_passive(self.model, self.data, show_left_ui= not disable_panels, show_right_ui=not disable_panels)
    
    def _sync_view(self):
        if self.view and self.view.is_running():
            with self.view.lock():
                self.view.sync()
    
    def _get_obs(self):
        robotic_arm_pointer = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')]
        target = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')]
        return np.concatenate([target, target - robotic_arm_pointer, self.data.qpos, self.data.qvel])
    
    def _compute_reward(self, action):

        # Calculate the distance
        distance = self._compute_distance()

        # calculate the action norm
        action_norm = np.linalg.norm(action)

        return -distance - action_norm # Extra penalty for doing drastic actions
    
    def _is_done(self):
        # robotic_arm_pointer = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')]
        # target = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')]
        # distance = np.linalg.norm(target - robotic_arm_pointer)
        return False
    def _is_out_of_bounds(self, robotic_arm_pointer) -> bool:
        return not ( (0.1 < robotic_arm_pointer[2]  < 0.9) and\
                     (-0.4 < robotic_arm_pointer[1] < 0.4) and\
                     (0.1 < robotic_arm_pointer[0] < 0.9)) 
    def _compute_distance(self):
        robotic_arm_pointer = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')]
        target = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'spherical_site')]
        return np.linalg.norm(target - robotic_arm_pointer)
    

def test_env():
    env = FrankaPandaEnv(render=True)
    obs = env.reset()

    while True:
        # pass
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        time.sleep(0.2)
        if done:
            print(obs, reward, done)
            _ = env.reset()
            
        
def run_manual():
    
    env = FrankaPandaEnv(render=True, manual_mode=True)
    # obs = env.reset()
    while env.view.is_running():
        mujoco.mj_step(env.model, env.data, 20)

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
            
    



