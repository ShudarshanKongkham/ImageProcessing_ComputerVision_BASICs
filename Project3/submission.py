import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
import threading

# Set up absolute paths for YOLO files
current_folder = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_folder, "yolov3.weights")
cfg_path = os.path.join(current_folder, "yolov3.cfg")
coco_path = os.path.join(current_folder, "coco.names")
file_path = os.path.join(current_folder, "soccer-ball.mp4")

# Initialize YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open(coco_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to create a tracker by name
def createTrackerByName(trackerType):
    if trackerType == "MIL":
        tracker = cv2.TrackerMIL_create()
    elif trackerType == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif trackerType == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        print("MIL, KCF, CSRT")
    return tracker

# Function to detect objects using YOLO
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "sports ball":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    indexes = indexes.flatten() if len(indexes) > 0 else []
    return boxes, indexes, class_ids

# Function to draw bounding box
def draw_bounding_box(frame, box, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

# Function to process video
def process_video(video_path, tracker_name):
    tracker = createTrackerByName(tracker_name)
    cap = cv2.VideoCapture(video_path)
    tracking = False
    detected_class = "None"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if tracking:
            success, box = tracker.update(frame)
            if success:
                draw_bounding_box(frame, box, (0, 255, 0))  # Green for tracking
                cv2.putText(frame, f"{tracker_name} Tracker", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Status: Tracking", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                tracking = False
                cv2.putText(frame, f"{tracker_name} Tracker", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Status: Tracking Failed", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not tracking:
            boxes, indexes, class_ids = detect_objects(frame)
            if len(indexes) > 0:
                box = boxes[indexes[0]]
                tracker = createTrackerByName(tracker_name)  # Reinitialize the tracker
                tracker.init(frame, tuple(box))
                tracking = True
                draw_bounding_box(frame, box, (255, 0, 0))  # Blue for detection
                detected_class = classes[class_ids[indexes[0]]]
            else:
                detected_class = "None"
        
        cv2.putText(frame, f"Detection: {detected_class}", (frame.shape[1] - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to start video processing with selected tracker
def start_processing():
    tracker_name = tracker_var.get()
    threading.Thread(target=process_video, args=(file_path, tracker_name), daemon=True).start()

# Create GUI for tracker selection
root = tk.Tk()
root.title("Tracker Selection")

tracker_var = tk.StringVar(value="CSRT")
tracker_label = tk.Label(root, text="Select Tracker:")
tracker_label.pack(pady=10)
tracker_dropdown = ttk.Combobox(root, textvariable=tracker_var)
tracker_dropdown['values'] = ("MIL", "KCF", "CSRT")
tracker_dropdown.pack(pady=10)

start_button = tk.Button(root, text="Start", command=start_processing)
start_button.pack(pady=20)

root.mainloop()
