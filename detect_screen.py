import cv2
import numpy as np
from ultralytics import YOLO
import mss

# ✅ Load your trained model (replace with your best model path)
model = YOLO(r"runs\train\smoke_yolov8\weights\best.pt")

# Define screen capture region
# You can change 'mon' if you have multiple monitors
monitor = {"top": 300, "left": 300, "width": 1280, "height": 720}

# Initialize mss for screen capture
sct = mss.mss()

while True:
    # Capture the screen
    frame = np.array(sct.grab(monitor))

    # Convert BGRA → BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Run YOLO inference on the frame
    results = model.predict(source=frame, conf=0.4, imgsz=640, device=0, show=False, verbose=False)

    # Draw detection boxes directly on frame
    annotated_frame = results[0].plot()

    # Display result
    cv2.imshow("Smoke Detection (Screen Capture)", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
