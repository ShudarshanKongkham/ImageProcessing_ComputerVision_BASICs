import cv2
import numpy as np

# Load the image
image_path = 'blemish.png'
image = cv2.imread(image_path)
original_image = image.copy()

# Set the blemish patch radius
blemish_radius = 15  # Radius of the patch to be used for blemish removal

# Blemish removal function
def remove_blemish(event, x, y, flags, param):
    global image, original_image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert the image to grayscale for blemish detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian filter to detect edges (blemishes)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)

        # Threshold the Laplacian result to create a binary mask of potential blemish regions
        _, mask = cv2.threshold(laplacian_abs, 50, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        #  Patch Selection: Find the nearest non-blemish pixel
        for i in range(y - blemish_radius, y + blemish_radius + 1):
            for j in range(x - blemish_radius, x + blemish_radius + 1):
                if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and mask[i, j] == 0:
                    # Extract the patch around the non-blemish pixel
                    patch = image[i - blemish_radius:i + blemish_radius + 1, j - blemish_radius:j + blemish_radius + 1]
                    break
            if 'patch' in locals():
                break

        # Patch Blending using seamless cloning
        if 'patch' in locals():
            center = (x, y)
            # Create a circular mask for the patch
            mask_patch = np.zeros((blemish_radius * 2 + 1, blemish_radius * 2 + 1), dtype=np.uint8)
            cv2.circle(mask_patch, (blemish_radius, blemish_radius), blemish_radius, (255, 255, 255), -1)
            # Blur the patch to smooth the edges
            patch_blurred = cv2.GaussianBlur(patch, (5, 5), 0)
            # Apply seamless cloning to blend the patch into the image
            image = cv2.seamlessClone(patch_blurred, image, mask_patch, center, cv2.NORMAL_CLONE)

        # Display the updated image
        cv2.imshow('Blemish Removal', image)

# Undo function to revert to the original image
def undo_last_operation():
    global image, original_image
    image = original_image.copy()
    cv2.imshow('Blemish Removal', image)

# Display the image and set mouse callback for blemish removal
cv2.imshow('Blemish Removal', image)
cv2.setMouseCallback('Blemish Removal', remove_blemish)

# Print instructions for the user
print(f"Using a patch of radius {blemish_radius} for blemish removal. Press 'u' to undo the last operation or 'q' to quit.")

# Main loop to handle user input
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u'):
        undo_last_operation()
    elif key == ord('q'):
        break

# Destroy all OpenCV windows
cv2.destroyAllWindows()