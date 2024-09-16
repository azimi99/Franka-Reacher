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


def image_render(env, cam_name="cam"):
    env.renderer.update_scene(env.data, camera=cam_name)
    image = None
    image = env.renderer.render()
    return image


    



