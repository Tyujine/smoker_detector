# detect_video.py
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs/train/smoke_yolov8/weights/best.pt"  # adjust if different
CONF_THRESH = 0.25
IOU_THRESH = 0.45

def draw_boxes(frame, results):
    h, w = frame.shape[:2]
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box[-1])
            conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box[4])
            if conf < CONF_THRESH: 
                continue
            xyxy = box.xyxy[0].cpu().numpy()  # (x1,y1,x2,y2)
            x1,y1,x2,y2 = map(int, xyxy)
            label = f"smoke {conf:.2f}"
            color = (0,0,255)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            # put label
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return frame

def main():
    model = YOLO(MODEL_PATH)
    # choose source: 0 for webcam or path to video file
    cap = cv2.VideoCapture(0)  # or "test.mp4"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # run inference (Ultralytics returns results list)
        res = model(frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)  # list-like
        # draw
        out = draw_boxes(frame, res)
        cv2.imshow("Smoke Detection", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
