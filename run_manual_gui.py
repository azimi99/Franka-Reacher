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
from  matplotlib import pyplot as plt
import threading

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



def visualize_plot(plot_sm_name):
    fig, ax = plt.subplots()

    plot_sm = shared_memory.SharedMemory(name=plot_sm_name)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    target_plot, = ax.plot([], [], marker='*', markersize=10, color='b', label='Target')
    end_effector_plot, = ax.plot([], [], marker='o', markersize=10, color='r', label='End Effector')
    # cam_capture = cv2.VideoCapture(0)
    # height 480, width 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter("manual_capture.mp4", fourcc, 30, (1280, 480))
    while True:
        coordinates = np.ndarray((6,), dtype=np.float32, buffer=plot_sm.buf)

        target_x_y = coordinates[:2]
        end_effector_x_y = coordinates[:3] - coordinates[3:]

        target_plot.set_data([target_x_y[0]], [target_x_y[1]])
        # Update the position of the end effector
        end_effector_plot.set_data([end_effector_x_y[0]], [end_effector_x_y[1]])
        end_effector_plot.set_markersize(5 + np.abs(coordinates[-1])*500)
        fig.canvas.draw()
        plt.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert the RGB image to BGR (OpenCV format)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # ret, frame = cam_capture.read()
        # if not ret:
        #     continue
        # height = min(img.shape[0], frame.shape[0])
        # img1_resized = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
        # img2_resized = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))

        # Horizontally stack the images
        # combined_img = np.hstack((img1_resized, img2_resized))
        # print(combined_img.shape)
        #Show the plot in the OpenCV window
        # cv2.imshow("Real-time Plot", combined_img)

        cv2.imshow("Visualization Plot", img)
        video_writer.write(img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 27 is the ASCII code for the 'ESC' key
            print("Escape key pressed. Closing the window.")
            break  # Exit the loop and close the window
    # cam_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()


def run_manual():
    print(tk.TkVersion)
    size = np.dtype(np.float32).itemsize * 8
    action_sm = shared_memory.SharedMemory(create=True, size=size)
    action = np.ndarray((8,), dtype=np.float32, buffer=action_sm.buf)
    reset_sm = shared_memory.SharedMemory(create=True, size=1)
    plot_sm = shared_memory.SharedMemory(create=True, size=np.dtype(np.float32).itemsize*6)
    coordinates = np.ndarray((6,), dtype=np.float32, buffer=plot_sm.buf)
    lock = Lock()

    for i in range(8):
        action[i] = 0.0

    env = FrankaPandaEnv(render=True)
    _ = env.reset()
    time.sleep(0.2)

    gui_process = Process(target=start_gui, args=(action_sm.name, reset_sm.name))
    gui_process.start()
    print(plot_sm.name)
    plot_process = Process(target=visualize_plot, args=(plot_sm.name,))
    plot_process.start()
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
        coordinates[:6] = obs[:6]
        

        step += 1



    env.close() 
    gui_process.join()
    plot_process.join()
    plot_sm.close()
    plt.close()
    action_sm.close()



if __name__ == "__main__":


    run_manual()
