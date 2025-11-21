from ultralytics import YOLO
import os

DATASET_PATH = 'dataset_augmented/'
DATA_YAML = DATASET_PATH + 'data.yaml'
MODEL_NAME = 'yolo11s.pt'
SAVE_DIR = 'runs/train/'

def train():
    model = YOLO('runs/train/smokerdetector_stage_A/weights/last.pt')
    model.train(data=DATA_YAML,
                epochs=50,
                imgsz=640,
                batch=16,
                device=0,
                pretrained=True,
                name='smokerdetector_stage_A',
                project=SAVE_DIR,
                patience=20,
                )

if __name__ == "__main__":
    train()