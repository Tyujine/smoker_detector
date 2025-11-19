from ultralytics import YOLO
import os

DATASET_PATH = 'dataset/Smoker Detector-New/'
DATA_YAML = DATASET_PATH + 'data.yaml'
MODEL_NAME = 'yolo11l.pt'
SAVE_DIR = 'runs/train/'

pretrained_weights = SAVE_DIR + 'smokerdetector_stage_2/weights/best.pt'

model = YOLO(MODEL_NAME)   # even larger

model.train(
    data="data.yaml",
    epochs=25,
    imgsz=1024,
    batch=4,
    device=0,
    pretrained=pretrained_weights,
    close_mosaic=10,
    device=0,
    name='smokerdetector_stage_3',
    project=SAVE_DIR
)