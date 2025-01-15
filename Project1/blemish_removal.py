import cv2
import numpy as np

# Load the image
image_path = 'blemish.png'
image = cv2.imread(image_path)
original_image = image.copy()

# Function to remove blemish using inpainting
def remove_blemish(event, x, y, flags, param):
    global image, original_image
    if event == cv2.EVENT_LBUTTONDOWN:
        blemish_size = 9  # Size of the blemish region

        # Define the mask for the blemish region
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(mask, (x, y), blemish_size, (255, 255, 255), -1)

        # Visualize the mask
        # cv2.imshow('Mask', mask)

        # Perform inpainting
        image = cv2.inpaint(image, mask, 2, cv2.INPAINT_TELEA)

        # Display the updated image
        cv2.imshow('Blemish Removal', image)

# Function to undo the last operation
def undo_last_operation():
    global image, original_image
    image = original_image.copy()
    cv2.imshow('Blemish Removal', image)

# Display the image
cv2.imshow('Blemish Removal', image)
cv2.setMouseCallback('Blemish Removal', remove_blemish)

# Wait for user input
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u'):  # Press 'u' to undo the last operation
        undo_last_operation()
    elif key == ord('q'):  # Press 'q' to quit the application
        break

cv2.destroyAllWindows()