import cv2
import numpy as np

def combine_images(frame, img, axis=1):
    """
    Combines a captured frame and an image either horizontally or vertically.

    Parameters:
    - frame: The captured frame from the webcam.
    - img: The loaded image to place next to the frame.
    - axis: 1 for horizontal, 0 for vertical.

    Returns:
    - combined_image: The combined image (NumPy array).
    """
    # Resize the image to match the height (for horizontal stacking) or width (for vertical stacking)
    if axis == 1:  # Horizontal stacking
        img = cv2.resize(img, (img.shape[1], frame.shape[0]))
    else:  # Vertical stacking
        frame = cv2.resize(frame, (img.shape[1], frame.shape[0]))

    # Combine both images along the given axis
    combined_image = np.concatenate((frame, img), axis=axis)
    
    return combined_image



# Example usage
output_filename = 'combined_video.mp4'  # Output video file
img_path = 'test.jpg'    # Path to the image to place next to the frame
duration = 10  # Duration of the video in seconds
fps = 30  # Frames per second
cap = cv2.VideoCapture(0)
    # Combine the first frame with the image to determine output dimensions
if not cap.isOpened():
    exit(0)
_, frame = cap.read()
height, width, _ = frame.shape

    # Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Calculate the number of frames to capture based on the desired duration
num_frames = int(duration * fps)

# Capture frames and write the combined images to the video
while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Write the combined image to the video file
    # video_writer.write(frame)

    # Optionally display the combined image
    cv2.imshow('Combined Image', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
video_writer.release()