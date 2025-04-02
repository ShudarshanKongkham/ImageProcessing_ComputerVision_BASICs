import cv2
import numpy as np

def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Apply Canny edge detection
    edged_image = cv2.Canny(blurred_image, 75, 200)
    return edged_image

def find_contours(edged_image):
    # Find contours
    contours, _ = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area and keep the largest one
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    return sorted_contours

def get_document_contour(sorted_contours):
    for contour in sorted_contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # If our approximated contour has four points, we can assume we have found the document
        if len(approx_contour) == 4:
            return approx_contour
    return None

def perspective_transform(image, document_contour):
    # Obtain a consistent order of the points and unpack them individually
    points = document_contour.reshape(4, 2)
    rectangle = np.zeros((4, 2), dtype="float32")

    sum_points = points.sum(axis=1)
    rectangle[0] = points[np.argmin(sum_points)]
    rectangle[2] = points[np.argmax(sum_points)]

    diff_points = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(diff_points)]
    rectangle[3] = points[np.argmax(diff_points)]

    # Compute the width and height of the new image
    (top_left, top_right, bottom_right, bottom_left) = rectangle
    width_A = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_B = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))

    max_width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_B = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    max_height = max(int(height_A), int(height_B))

    # Set of destination points for "birds eye view"
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rectangle, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


# Path to the image
image_path = "scanned-form.jpg"

# Load the image
image = cv2.imread(image_path)
# Preprocess the image
edged_image = preprocess_image(image)
# Find contours
sorted_contours = find_contours(edged_image)
cv2.imshow("Contours", cv2.drawContours(image.copy(), sorted_contours, -1, (0, 255, 0), 2))

# Get the document contour
document_contour = get_document_contour(sorted_contours)
if document_contour is None:
    raise Exception("Could not find document contour.")

# Draw the document contour
cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 2)
cv2.imshow("Document Contour", image)

# Apply perspective transform
warped_image = perspective_transform(image, document_contour)

# Check the output size and set the output width to 500 pixels
output_width = 500
aspect_ratio = output_width / warped_image.shape[1]
output_height = int(warped_image.shape[0] * aspect_ratio)
warped_image = cv2.resize(warped_image, (output_width, output_height))

# Save the warped image
cv2.imwrite("scanned_document.jpg", warped_image)

# Display the warped images
cv2.imshow("Scanned Document, width=500px", warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
