import cv2
import numpy as np

def on_trackbar_change(val):
    # No-op to avoid race conditions
    pass

def process_frame():
    global frame, mask, background, output

    # Convert selected color to NumPy array
    selected_color_np = np.array(selected_color, dtype=np.uint8)
    
    # Adjust tolerance in BGR space and clip to valid range
    lower_green = np.clip(selected_color_np - tolerance, 0, 255)
    upper_green = np.clip(selected_color_np + tolerance, 0, 255)
    
    # Create mask based on the selected color and tolerance
    mask = cv2.inRange(frame, lower_green, upper_green)

    # Apply softness/feathering to the mask
    if softness > 0:
        mask = cv2.GaussianBlur(mask, (int(max(softness, 0) * 21) | 1, int(max(softness, 0) * 21) | 1), 0)

    # Apply color cast removal
    if color_cast > 0:
        frame = cv2.addWeighted(frame, 1 - color_cast, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), color_cast, 0)

    # Apply hue shift
    if hue > 0:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame[..., 0] = (hsv_frame[..., 0].astype(int) + hue) % 180
        frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

    # Apply defringe (morphological close operation on the mask)
    if defringe > 0:
        k_size = (defringe * 2) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Combine foreground and background
    fg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    bg = cv2.bitwise_and(background, background, mask=mask)
    output = cv2.add(fg, bg)

    # Display the output
    cv2.imshow('Output', output)

def select_color(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        # Select color from the frame at the clicked position
        selected_color = frame[y, x].tolist()
        print("Selected color:", selected_color)
        process_frame()

# --- Main Program ---

# Load video and background
# video_path = 'greenscreen-asteroid.mp4'  #video file path
video_path = 'greenscreen-demo.mp4'  #video file path

background_path = 'nightsky.jpeg'  # path to background image
cap = cv2.VideoCapture(video_path)
background = cv2.imread(background_path)
background = cv2.resize(background, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Initialize variables
selected_color = [0, 0, 0]  # Initial color (black)
tolerance = 0
softness = 0.0
color_cast = 0.0
hue = 0
defringe = 0

# Create windows
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', 800, 600)

# Create trackbars with no-op callback
cv2.createTrackbar('Tolerance', 'Output', tolerance, 100, on_trackbar_change)
cv2.createTrackbar('Softness', 'Output', int(softness * 100), 100, on_trackbar_change)
cv2.createTrackbar('Color Cast', 'Output', int(color_cast * 100), 100, on_trackbar_change)
cv2.createTrackbar('Hue', 'Output', hue, 179, on_trackbar_change)
cv2.createTrackbar('Defringe', 'Output', defringe, 10, on_trackbar_change)

# Set mouse callback for color selection
cv2.setMouseCallback('Output', select_color)

# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        # Loop the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break

    # Read trackbar values
    tolerance = cv2.getTrackbarPos('Tolerance', 'Output')
    softness = cv2.getTrackbarPos('Softness', 'Output') / 100.0
    color_cast = cv2.getTrackbarPos('Color Cast', 'Output') / 100.0
    hue = cv2.getTrackbarPos('Hue', 'Output')
    defringe = cv2.getTrackbarPos('Defringe', 'Output')
    process_frame()

    # Draw a small rectangle indicating the selected color
    cv2.rectangle(output, (10, 10), (60, 60), (int(selected_color[0]), int(selected_color[1]), int(selected_color[2])), -1)
    cv2.putText(output, "Sel", (13, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Output', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()