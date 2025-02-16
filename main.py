import cv2 
import torch
from ultralytics import YOLO

# Load the model (choose the appropriate model)
model_path = r'yolov8football-20250216T051506Z-001\yolov8football\yolov8m-football.pt'  # Change to 'yolov8s-football.pt' if needed
model = YOLO(model_path)

# Input video path
video_path = r'football.mp4'

# Output video path
output_path = 'output.mp4'

# Set detection parameters
conf_threshold = 0.25
img_size = 1280

# Open the input video
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Set up the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

# Define specific colors for each class
class_colors = {
    'ball': (255, 165, 0),        # Yellow for ball
    'player': (255, 20, 147),      # Dark pink for player
    'referee': (50, 205, 50)        # Dark green for referee
}

def get_class_color(class_name):
    return class_colors.get(class_name, (255, 255, 255))  # Default to white if not specified

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(frame, conf=conf_threshold, imgsz=img_size)

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = f"{result.names[class_id]}"

            # Get color for class
            color = get_class_color(label.lower())

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow('Football Detection', frame)

    # Write the frame to the output video
    out.write(frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
