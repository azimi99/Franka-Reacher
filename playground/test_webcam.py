import cv2
import time
import numpy as np


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit(0)

        
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
fps = 30

ret, frame = cap.read()
height, width, _ = frame.shape

video_writer = cv2.VideoWriter("test.mp4", fourcc, fps, (width, height))

# Calculate the number of frames to capture based on the desired duration


# Capture frames and write the combined images to the video
while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Write the combined image to the video file
    video_writer.write(frame)

    # Optionally display the combined image
    cv2.imshow('Combined Image', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
video_writer.release()
