from ultralytics import YOLO
import os

DATASET_PATH = 'dataset_augmented/'
DATA_YAML = DATASET_PATH + 'data.yaml'
MODEL_NAME = 'yolo11l.pt'
SAVE_DIR = 'runs/train/'

pretrained_weights = SAVE_DIR + 'smokerdetector_stage_A/weights/best.pt'

def train():
    model = YOLO(MODEL_NAME)
    model.load(pretrained_weights)
    model.train(data=DATA_YAML,
                epochs=100,
                imgsz=832,
                batch=8,
                device=0,
                name='smokerdetector_stage_B',
                project=SAVE_DIR,
                lr0=0.001,
                lrf=0.0001,
                mosaic=0.0,
                )

if __name__ == "__main__":
    train()