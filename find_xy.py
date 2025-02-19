import cv2

# Load your image (replace with your image path)
image_path = "real_camera_image.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
clone = image.copy()
points = {}

# A counter for naming points (e.g., p1, p2, ...)
point_counter = 1

def click_event(event, x, y, flags, param):
    global clone, point_counter
    # On left-click, record the point and draw a circle.
    if event == cv2.EVENT_LBUTTONDOWN:
        point_name = f'p{point_counter}'
        points[point_name] = (x, y)
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(clone, point_name, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2)
        cv2.imshow("Image", clone)
        print(f"Recorded {point_name}: ({x}, {y})")
        point_counter += 1
    # Right-click to clear all points and reset the image.
    elif event == cv2.EVENT_RBUTTONDOWN:
        clone = image.copy()
        points.clear()
        print("Cleared all points.")
        cv2.imshow("Image", clone)

# Create a window and set the mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_event)

print("Left-click to record a point. Right-click to clear all points. Press 'q' to exit.")

while True:
    cv2.imshow("Image", clone)
    key = cv2.waitKey(1) & 0xFF
    # Exit on 'q' key
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("Final expected pixel points:", points)
