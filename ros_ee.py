#!/home/chemist/Desktop/sim2real/franka/bin/python
import os
import time
import math
import numpy as np
import cv2
import rospy
import tf
import pyrealsense2 as rs

os.environ['ROS_MASTER_URI'] = 'http://172.16.0.1:11311'
os.environ['ROS_IP'] = '172.16.0.3'

def get_end_effector_transform():
    rospy.init_node('ee_tf_listener', anonymous=True)
    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            listener.waitForTransform('/world', '/panda_EE', rospy.Time(0), rospy.Duration(4.0))
            (trans, rot) = listener.lookupTransform('/world', '/panda_EE', rospy.Time(0))
            print("EE Translation:", trans)
            print("EE Rotation (quaternion):", rot)
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("TF Exception:", e)
            rate.sleep()
    return None, None

def compute_camera_matrix(ee_pos, ee_quat, fx, fy, cx, cy):
    # offset = np.array([0.047, -0.028, 0.049])
    offset = np.array([0.047, -0.028, 0.049])
    print("Offset:", offset)
    cam_pos = np.array(ee_pos) + offset
    print("EE Position:", ee_pos)
    print("Camera Position:", cam_pos)
    rpy = tf.transformations.euler_from_quaternion(ee_quat)
    
    print("EE Euler angles (radians):", rpy)
    rpy = (rpy[0], rpy[1], rpy[2])



    print("Camera Euler angles (radians):", rpy)
    cam_quat = tf.transformations.quaternion_from_euler(rpy[0] + math.radians(-2), rpy[1] + math.radians(7), rpy[2] + math.radians(91))
    print("Camera Quaternion:", cam_quat)
    R = tf.transformations.quaternion_matrix(cam_quat)[0:3, 0:3]
    print("Rotation Matrix R:", R)
    t = -np.dot(R, cam_pos.reshape(3,1))
    print("Translation vector t:", t)
    extrinsic = np.hstack((R, t))
    print("Extrinsic matrix:", extrinsic)
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    print("Intrinsic matrix K:", K)
    P = K @ extrinsic
    print("Camera Projection Matrix P:", P)
    return P

def world_2_pixel(world_coordinate, camera_matrix):
    w = np.ones((4,), dtype=float)
    w[0:3] = world_coordinate
    result = camera_matrix @ w
    xs, ys, s = result
    x = xs / s
    y = ys / s

    return int(round(x)), int(round(y))

def main():
    ee_trans, ee_rot = get_end_effector_transform()
    if ee_trans is None:
        print("No EE transform received.")
        return

    fx = 645.041
    fy = 645.041
    cx = 651.277
    cy = 361.249

    print("Camera intrinsics: fx={}, fy={}, cx={}, cy={}".format(fx, fy, cx, cy))
    P = compute_camera_matrix(ee_trans, ee_rot, fx, fy, cx, cy)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    pipeline.start(config)
    frames = None
    start_time = time.time()
    while time.time() - start_time < 10:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            if frames:
                break
        except RuntimeError as e:
            print("RuntimeError while waiting for frames:", e)
            pipeline.stop()
            time.sleep(1)
            pipeline.start(config)
    if not frames:
        print("No frames received.")
        pipeline.stop()
        return
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("No color frame received.")
        pipeline.stop()
        return
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("real_camera_image.png", color_image)
    points = {
        'p1': [0.755, -0.378, 0.021],
        'p2': [0.757, 0.363, 0.020],
        'p3': [0.330, 0.381, 0.015],
        'p4': [0.320, -0.393, 0.015]
    }
    for key, pt in points.items():
        pixel = world_2_pixel(pt, P)
        print("Point", key, ":", pt, "-> Pixel:", pixel)
        cv2.circle(color_image, pixel, 5, (0, 0, 255), -1)
    cv2.imshow("Image", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == '__main__':
    main()