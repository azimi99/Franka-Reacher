from real_non_visual_reacher import FrankaPandaEnv
import time
import numpy as np 
from multiprocessing import shared_memory, Process, Queue, Lock


import cv2


np.set_printoptions(precision=3, linewidth=10000, suppress=True)
import os
import json
import time
import tkinter as tk
from tkinter import ttk
import argparse

from utils import image_render, combine_images




class JointControlApp(tk.Tk):
    def __init__(self, sm_name, reset_sm_name):
        super().__init__()
        self.title("7-Joint Control GUI")
        self.geometry("1200x1800")

        self.action_sm = shared_memory.SharedMemory(name=sm_name)
        self.action = np.ndarray((8,), dtype=np.float32, buffer=self.action_sm.buf)
        self.reset_sm = shared_memory.SharedMemory(name=reset_sm_name)
        self.font1 = ("Helvetica", 14)
        self.font2 = ("Helvetica", 14, "bold")
        self.font3 = ("Helvetica", 16, "bold")

        self.create_widgets() 

    def create_widgets(self):
        for i in range(7):
            frame = ttk.Frame(self)
            frame.pack(pady=20)

            label = ttk.Label(frame, text=f"Joint {i+1}", font=self.font1)
            label.pack(side=tk.LEFT, padx=10)

            btn_minus = tk.Button(frame, text="-", width=8, height=3, font=self.font2)
            btn_minus.pack(side=tk.LEFT, padx=(5, 10))
            btn_minus.bind("<ButtonPress>", lambda e, j=i: self.on_press_minus(j))
            btn_minus.bind("<ButtonRelease>", lambda e, j=i: self.on_release_minus(j))

            btn_plus = tk.Button(frame, text="+", width=8, height=3, font=self.font2)
            btn_plus.pack(side=tk.LEFT, padx=(10, 10))
            btn_plus.bind("<ButtonPress>", lambda e, j=i: self.on_press_plus(j))
            btn_plus.bind("<ButtonRelease>", lambda e, j=i: self.on_release_plus(j))


        open_button = tk.Button(self, text="Open", width=16, height=4, font=self.font2)
        open_button.pack(pady=20)
        open_button.bind("<ButtonPress>", self.on_press_open)

        reset_button = tk.Button(self, text="RESET", width=16, height=4, font=self.font2)
        reset_button.pack(pady=20)
        reset_button.bind("<ButtonPress>", self.on_press_reset)

        close_button = tk.Button(self, text="Close", width=16, height=4, font=self.font2)
        close_button.pack(pady=20)
        close_button.bind("<ButtonPress>", self.on_press_close)

        stop_button = tk.Button(self, text="STOP", width=20, height=6, font=self.font3)
        stop_button.pack(pady=40, ipadx=40, ipady=20)
        stop_button.bind("<ButtonPress>", self.on_press_stop)

    def update_joint_position(self, joint, change):
        self.action[joint] = change

    def on_press_minus(self, joint):
        self.update_joint_position(joint, -0.3)

    def on_release_minus(self, joint):
        self.update_joint_position(joint, 0)
    
    def on_press_reset(self, event):
        print("Resetting Arm")
        self.reset_sm.buf[0] = True

    def on_press_plus(self, joint):
        self.update_joint_position(joint, 0.3)

    def on_release_plus(self, joint):
        self.update_joint_position(joint, 0)

    def on_press_open(self, event):
        self.action[7] = 1.1

    def on_press_close(self, event):
        self.action[7] = -1.1

    def on_press_stop(self, event): 
        for i in range(8):
            self.action[i] = -2.0
        self._close()

    def _close(self):
        self.action_sm.close()
        self.destroy()


def start_gui(sm_name, reset_sm_name):
    app = JointControlApp(sm_name, reset_sm_name)
    app.mainloop()



def get_tf_mat(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta

    return np.array([[np.cos(q), -np.sin(q), 0, a],
                     [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                     [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                     [0, 0, 0, 1]])


def run_manual():
    print(tk.TkVersion)
    size = np.dtype(np.float32).itemsize * 8
    action_sm = shared_memory.SharedMemory(create=True, size=size)
    action = np.ndarray((8,), dtype=np.float32, buffer=action_sm.buf)
    reset_sm = shared_memory.SharedMemory(create=True, size=1)
    lock = Lock()

    for i in range(8):
        action[i] = 0.0

    env = FrankaPandaEnv(render=True)
    _ = env.reset()
    time.sleep(0.2)

    gui_process = Process(target=start_gui, args=(action_sm.name, reset_sm.name))
    gui_process.start()


    poses = {}
    step = 0

    while True:

        if reset_sm.buf[0]:
            env.reset()
            time.sleep(0.5)
            reset_sm.buf[0] = False
        a = action.copy()
        if a[0] < -1.0:
            break

        if a[7] > 1.0:
            print('Openning gripper')
            action[7] = 0.0
            env.open_gripper()

        if a[7] < -1.0:
            print('Closing gripper')
            action[7] = 0.0
            env.close_gripper()
        
        
        # time.sleep(0.04)
        
        obs, reward, done, info = env.step(a)

        step += 1



    env.close() 
    gui_process.join()

    action_sm.close()



if __name__ == "__main__":


    run_manual()
