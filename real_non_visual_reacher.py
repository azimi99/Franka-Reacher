from tkinter import S
import numpy as np

import time
from gymnasium import spaces

import gymnasium as gym
from gymnasium.core import ActionWrapper
import numpy as np
from gymnasium import spaces
import os

from numpy.core.defchararray import count
import rospy
from PIL import Image
import math
from collections import deque

from franka_utils import *

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import time
import logging
import cv2

from franka_interface import ArmInterface, RobotEnable, GripperInterface
# ids camera lib for use of IDS ueye cameras.
# https://www.ids-imaging.us/files/downloads/ids-peak/readme/ids-peak-linux-readme-1.2_EN.html
#import ids
import time
import signal
import multiprocessing
from gymnasium.spaces import Box as GymBox

import mujoco
import mujoco.viewer

ARM_VEL_LIMITS = np.array([2.61799, 2.61799, 2.61799, 2.61799, 3.14159, 3.14159, 3.14159, 0])

Q_MAX = np.array([2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973, 0])


class FrankaPandaEnv(gym.Env):

    """
    Gym env for the real franka robot. Set up to perform the placement of a peg that starts in the robots hand into a slot
    """
    def __init__(self, dt=0.04, episode_length=8, camera_index=0, seed=9, size_tol=0.45, render=False, model_path="env.xml"):
        np.random.seed(seed)
        self.DT= dt
        self.dt = dt
        self.ep_time = 0
        self.max_episode_duration = episode_length # in seconds
        signal.signal(signal.SIGINT, self.exit_handler)
        # config_file = os.path.join(os.path.dirname(__file__), os.pardir, 'reacher.yaml')
        self.configs = configure('./reacher.yaml')
        self.conf_exp = self.configs['experiment']
        self.conf_env = self.configs['environment']
        rospy.init_node("franka_robot_gym")
        self.init_joints_bound = self.conf_env['reset-bound']
        #self.target_joints = self.conf_env['target-bound']
        self.safe_bound_box = np.array(self.conf_env['safe-bound-box'])
        self.target_box = np.array(self.conf_env['target-box'])
        self.joint_angle_bound = np.array(self.conf_env['joint-angle-bound'])
        self.return_point = self.conf_env['return-point']
        self.out_of_boundary_flag = False
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

        self.robot = ArmInterface(True)
        self.gripper = GripperInterface()
        force = 1e-6
        self.robot.set_collision_threshold(cartesian_forces=[force,force,force,force,force,force])
        self.robot.exit_control_mode(0.1)
        self.robot_status = RobotEnable()
        self.control_frequency = 1/dt
        self.rate = rospy.Rate(self.control_frequency)
        
        self._size_tol = size_tol

        self.ct = dt
        self.tv = time.time()


        self.joint_states_history = deque(np.zeros((5, 21)), maxlen=5)
        self.torque_history = deque(np.zeros((5, 7)), maxlen=5)
        self.last_action_history = deque(np.zeros((5, 7)), maxlen=5)
        self.time_out_reward = False
        action_dim = 7
        self.prev_action = np.zeros(action_dim)

        self.max_time_steps = int(self.max_episode_duration / dt)

        self.previous_place_down = None

        self.joint_action_limit = 0.3

        # self.action_space = GymBox(low=-self.joint_action_limit * np.ones(7), high=self.joint_action_limit*np.ones(7))
        # self.joint_angle_low = [j[0] for j in self.joint_angle_bound]
        # self.joint_angle_high = [j[1] for j in self.joint_angle_bound]

        # self.observation_space = GymBox(
        #     low=np.array(
        #         self.joint_angle_low  # q_actual
        #         + list(-np.ones(7)*self.joint_action_limit)  # qd_actual
        #         + list(-np.ones(7)*self.joint_action_limit)  # previous action in cont space
        #     ),
        #     high=np.array(
        #         self.joint_angle_high  # q_actual
        #         + list(np.ones(7)*self.joint_action_limit)  # qd_actual
        #         + list(np.ones(7)*self.joint_action_limit)    # previous action in cont space
        #     )
        # )
        ## Parallel real-time rendering of the virtual environment
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 3 + self.model.nq + self.model.nv,), dtype=np.float32)

        self.total_timesteps = 0


    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def monitor(self, reward, done, info):
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0
        return info
        
    def reset(self):
        """
        reset robot to random pose
        Returns
        -------
        object
            Observation of the current state of this env in the format described by this envs observation_space.
        """
        self.time_steps = 0
        self.ep_time = 0
        self.robot_status.enable()
        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))
        # close the gripper
        self.close_gripper()
        self._reset_stats()

        # set imaginary reacher-3d target
        # [0.72667744 0.16379048 0.20914678]
        self.target =\
              [ np.random.uniform(0.6, 0.7),
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(0.2, 0.6)] 

        # # random pose
        # random_reset_pose = [random_val_continous(joint_range) for joint_range in self.angle_safety_bound]
        # random_rest_joints = dict(zip(self.joint_names, random_reset_pose))
        # smoothly_move_to_position_vel(self.robot, self.robot_status, random_rest_joints ,MAX_JOINT_VELs=1.3)

        self.target_pose = [np.random.uniform(box_range[0], box_range[1]) for box_range in self.target_box]
        # # self.target_pose = [1.9, 0, -1.93, -1.52, 0.10, 1.52, 0.8]
        # self.target_pose[1] = 0  # set y to 0
        self.target_pose[2] = 0.5  # set z to string length
        self.reset_ee_quaternion = [0,-1.,0,0]
        
        obs = self.get_state()


        self.out_of_boundary_flag = False


        reset_pose = dict(zip(self.joint_names, [np.random.randint(-10, 10) / 100, -0.1, 0, -0.6, 0, 1.6, 0.8]))
        print(reset_pose)
        smoothly_move_to_position_vel(self.robot, self.robot_status, reset_pose, MAX_JOINT_VELs=1.3)
        # print(reset_pose)


        ## TODO: Update this
        reset_pose = dict(zip(self.joint_names, self.return_point))

        ## RANDOMIZE START POSITION -- SKIP JOINT 5 (SINCE IT MESSES WITH CAMERA ORIENTATION)
        reset_pose['panda_joint1'] = np.random.uniform(-0.3, 0.3)
        reset_pose['panda_joint5'] = np.random.uniform(-0.3, 0.3)
        # reset_pose['panda_joint7'] = np.random.uniform(-0.1, 0.1)


        smoothly_move_to_position_vel(self.robot, self.robot_status, reset_pose ,MAX_JOINT_VELs=1.3)
        # print("here", self.robot.endpoint_pose()["orientation"])
        print(reset_pose)

        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))

        # get the observation
        obs_robot = self.get_state()
        qpos = obs_robot["joints"].copy()
        qvel = obs_robot['joint_vels'].copy()

        obs = self._get_obs(qpos, qvel)

        self.time_steps = 0

        self.tv = time.time()
        self.reset_time = time.time()

        self.cur_step = 0 

        self._reset_stats()

        
        return obs.copy()


    def get_robot_jacobian(self):
        return self.robot.zero_jacobian()
 
    def euler_from_quaternion(self,q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w,x,y,z = q        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    def get_state(self):
        # get object state
        # self.obs_object = self.camera.get_state()
        
        # get robot states
        joint_angles = extract_values(self.robot.joint_angles(), self.joint_names)
        joint_velocitys = extract_values(self.robot.joint_velocities(), self.joint_names)
        # joint_efforts = extract_values(self.robot.joint_efforts(), self.joint_names)
        ee_pose = self.robot.endpoint_pose()
        ee_quaternion = [ee_pose['orientation'].w, ee_pose['orientation'].x,
                         ee_pose['orientation'].y, ee_pose['orientation'].z]

        self.last_action_history.append(self.prev_action)
        
        observation = {
            
            'last_action': self.prev_action,
            'joints': np.array(joint_angles),
            'joint_vels': np.array(joint_velocitys)
        }
        # print('orientation',ee_pose['orientation'])
        self.ee_position = ee_pose['position']
        # print(self.ee_position)
        self.ee_position_table = np.array([1.07-self.ee_position[0], 0.605-self.ee_position[1], self.ee_position[2]])
        self.ee_orientation = ee_quaternion
        #return observation['joints']
        return observation



    def out_of_boundaries(self):
        x, y, z = self.robot.endpoint_pose()['position']
        
        x_bound = self.safe_bound_box[0,:]
        y_bound= self.safe_bound_box[1,:]
        z_bound = self.safe_bound_box[2,:]
        if scalar_out_of_range(x, x_bound):
            # print('x out of bound, motion will be aborted! x {}'.format(x))
            return True
        if scalar_out_of_range(y, y_bound):
            # print('y out of bound, motion will be aborted! y {}'.format(y))
            return True
        if scalar_out_of_range(z, z_bound):
            # print('z out of bound, motion will be aborted!, z {}'.format(z))
            return True
        return False

    def apply_joint_vel(self, joint_vels):
        joint_vels = dict(zip(self.joint_names, joint_vels))

        
        self.robot.set_joint_velocities(joint_vels)        
        return True
    
    def apply_joint_displacement(self, displacement):
        observation_robot = self.get_state()

        ## Sync with simulator
        qpos = observation_robot["joints"].copy()



        # print(displacement.shape)

        new_qpos = qpos + displacement[:7]
                ## clip to within range
        new_qpos = np.clip(-Q_MAX[:7], Q_MAX[:7], new_qpos[:7])
        joint_pos = dict(zip(self.joint_names, new_qpos[:7]))
        # print(qpos.shape)
        # print(displacement.shape)
        self.robot.set_joint_positions(joint_pos)

    def step(self, action, pose_vel_limit=0.3):
        self.ep_time += self.dt
        self.cur_step += 1
        self.robot_status.enable()
        
        # limit joint action
        action = action.reshape(-1)
        action = np.clip(action, -1*ARM_VEL_LIMITS, ARM_VEL_LIMITS)
        # action = (((action + 1) * 0.5) * ARM_VEL_LIMITS * 2) - ARM_VEL_LIMITS

        # convert joint velocities to pose velocities
        pose_action = np.matmul(self.get_robot_jacobian(), action[:7])

        # limit action
        pose_action[:3] = np.clip(pose_action[:3], -pose_vel_limit, pose_vel_limit)

        # safety
        out_boundary = self.out_of_boundaries()
        pose_action[:3] = self.safe_actions(pose_action[:3])

        # calculate joint actions
        d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
        for i in range(3):
            if d_angle[i] < -np.pi:
                d_angle[i] += 2*np.pi
            elif d_angle[i] > np.pi:
                d_angle[i] -= 2*np.pi

        d_X = pose_action
        
        if out_boundary:
            d_X[3:] = 0
            action = self.get_joint_vel_from_pos_vel(d_X)

        action = self.handle_joint_angle_in_bound(action)

        # self.apply_joint_vel(action)
        self.apply_joint_displacement(action * self.dt/2)
        self.prev_action = action


        # pass time step duration

        done = False

        delay = (self.ep_time + self.reset_time) - time.time()
        if delay > 0:
            time.sleep(np.float64(delay))

        # get next observation
        observation_robot = self.get_state()

        self.time_steps += 1

        ## Sync with simulator
        qpos = observation_robot["joints"].copy()
        qvel = observation_robot["joint_vels"].copy()

        # construct the state
        obs = self._get_obs(qpos, qvel)
        prop = obs.copy()
        done = 0
        info = {}


        if self.ep_time >= (self.max_episode_duration-1e-3):
            done = True
            info['TimeLimit.truncated'] = True

        if done:
            self.apply_joint_vel(np.zeros((7,)))

        reward = self._compute_reward(self._compute_distance(observation_robot["joints"]), action)
        info = self.monitor(reward, done, info)

        return  prop, reward, done, info

    def _get_obs(self, joint_pos, joint_vels):
        joint_pos = np.append(joint_pos, [0.04, 0.04])
        joint_vels = np.append(joint_vels, [0.0, 0.0])
        robotic_arm_pointer = self._get_end_effector_pos(joint_angles=joint_pos)
        
        return np.concatenate([self.target, self.target - robotic_arm_pointer, joint_pos, joint_vels])

    def _compute_reward(self, distance, action):
        return -distance - np.linalg.norm(action)


    def handle_joint_angle_in_bound(self, action):
        current_joint_angle = self.robot.joint_angles()
        in_bound = [False] * 7
        for i, joint_name in enumerate(self.joint_names):
            if current_joint_angle[joint_name] > 0.05 + self.joint_angle_bound[i][1]:
                 
                action[i] = -0.5
            elif current_joint_angle[joint_name] < -0.05+ self.joint_angle_bound[i][0]:
                action[i] = +0.5
        return action

    def get_timeout_reward(self):
        if self.time_out_reward:
            reward = -1
            print('call time out reward {:+.3f}'.format(reward))
            return reward
        else:
            return 0

    def move_to_pose_ee(self, ref_ee_pos, pose_vel_limit=0.2):
        counter = 0
        # print('11111', rospy.Time.now())
        
        while True:
            self.robot_status.enable()
            # print(self.robot_status.state())
            counter += 1
            #action = agent.act(observations['ee_states'], ref_ee_pos, self.get_robot_jacobian(), add_noise=False)
            self.get_state()
            action = np.zeros((4,))
            action[:3] = ref_ee_pos-self.ee_position
            action[-1] = 1

            if max(np.abs(action[:3])) < 0.005 or counter > 100:
                break


            pose_action = np.clip(action[:3], -pose_vel_limit, pose_vel_limit)

            # calculate joint actions
            d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
            for i in range(3):
                if d_angle[i] < -np.pi:
                    d_angle[i] += 2*np.pi
                elif d_angle[i] > np.pi:
                    d_angle[i] -= 2*np.pi
            d_angle *= 0.5
            #print('d_angle', d_angle)
            d_X = np.array([pose_action[0], pose_action[1], pose_action[2], d_angle[0],d_angle[1],d_angle[2]])
            joints_action = self.get_joint_vel_from_pos_vel(d_X)
            # print('joints_action', joints_action)
            self.apply_joint_vel(joints_action)
            
            # action cycle time
            self.rate.sleep()
        self.apply_joint_vel(np.zeros((7,)))

    def get_joint_vel_from_pos_vel(self, pose_vel):
        return np.matmul(np.linalg.pinv( self.get_robot_jacobian() ), pose_vel)

    def safe_actions(self, action):
        out_boundary = self.out_of_boundaries()
        x, y, z = self.robot.endpoint_pose()['position']
        self.box_Normals = np.zeros((6,3))
        self.box_Normals[0,:] = [1,0,0]
        self.box_Normals[1,:] = [-1,0,0]
        self.box_Normals[2,:] = [0,1,0]
        self.box_Normals[3,:] = [0,-1,0]
        self.box_Normals[4,:] = [0,0,1]
        self.box_Normals[5,:] = [0,0,-1]
        self.planes_d = [   self.safe_bound_box[0][0],
                            -self.safe_bound_box[0][1],
                            self.safe_bound_box[1][0],
                            -self.safe_bound_box[1][1],
                            self.safe_bound_box[2][0],
                            -self.safe_bound_box[2][1]]
        if out_boundary:
            action = np.zeros((3,))
            for i in range(6):
                # action += 0.05 * self.box_Normals[i] * ( (self.box_Normals[i].dot(np.array([x,y,z])) - self.planes_d[i]) < 0 ) 
                ####
                action += 0.1 * self.box_Normals[i] * ( (self.box_Normals[i].dot(np.array([x,y,z])) - self.planes_d[i]) < 0 ) 

        return action

    def close(self):
        # stop the robot
        cv2.destroyAllWindows()

        self.apply_joint_vel(np.zeros((7,)))
    
    def exit_handler(self,signum):
        exit(signum)

    
    def terminate(self):
        self.close()
        self.exit_handler(1)

    def seed(self, seed):
        np.random.seed(seed)

    def open_gripper(self):
        return self.gripper.open()

    def close_gripper(self):
        return self.gripper.close()

    def _compute_distance(self, joint_angles):
        robotic_arm_pointer = self._get_end_effector_pos(joint_angles)
        target = np.array([0.7, 0.2, 0.3])
        return np.linalg.norm(target - robotic_arm_pointer)
    
    def _get_tf_mat(self, i, dh):
        a = dh[i][0]
        d = dh[i][1]
        alpha = dh[i][2]
        theta = dh[i][3]
        q = theta

        return np.array([[np.cos(q), -np.sin(q), 0, a],
                        [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                        [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                        [0, 0, 0, 1]])

    def _get_end_effector_pos(self, joint_angles):

        dh_params = [[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi/2, joint_angles[1]],
                 [0, 0.316, np.pi/2, joint_angles[2]],
                 [0.0825, 0, np.pi/2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi/2, joint_angles[4]],
                 [0, 0, np.pi/2, joint_angles[5]],
                 [0.088, 0, np.pi/2, joint_angles[6]],
                 [0, 0.107, 0, 0],
                 [0, 0, 0, -np.pi/4]]

        T = np.eye(4)
        for i in range(7 + 2):
            T = T @ self._get_tf_mat(i, dh_params)

        final_T = T @ self._get_end_effector_transformation()
        return final_T[:3, 3] # xyz position of end effector
    
    def _quaternion_to_rotation_matrix(self, quat):
        """
        Converts a quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix.
        """
        qx, qy, qz, qw = quat
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R
    
    def _get_end_effector_transformation(self):
        # End-effector position and quaternion from XML
        position = np.array([0, 0, 0.1034])  # Position from XML
        quat = [0.707, 0, 0.707, 0]          # Quaternion from XML
        rotation_matrix = self._quaternion_to_rotation_matrix(quat)

        # Construct the 4x4 transformation matrix
        T_end_effector = np.eye(4)
        T_end_effector[:3, :3] = rotation_matrix
        T_end_effector[:3, 3] = position
        return T_end_effector



