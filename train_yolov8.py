from ultralytics import YOLO
import os

dataset_path = '\dataset\Indoor Fire Smoke'
DATA_YAML = 'data.yaml'
MODEL_NAME = 'yolov8s.pt'
SAVE_DIR = 'runs/train/smoke_yolov8'

def train():
    model = YOLO(MODEL_NAME)
    model.train(data=DATA_YAML,
                epochs=50,
                imgsz=640,
                batch=8,
                workers=4,
                name='smoke_yolov8',
                project='runs/train',
                exist_ok=True)
    print('Training Finished')

if __name__ == "__main__":
    train()