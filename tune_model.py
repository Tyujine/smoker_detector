from ultralytics import YOLO
import os

DATASET_PATH = 'dataset/Smoker Detector-New/'
DATA_YAML = DATASET_PATH + 'data.yaml'
MODEL_NAME = 'yolo11m.pt'
SAVE_DIR = 'runs/train/'

pretrained_weights = SAVE_DIR + 'smokerdetector_stage_1/weights/best.pt'

def train():
    model = YOLO(MODEL_NAME)
    model.info()
    model.train(data=DATA_YAML,
                epochs=100,
                imgsz=832,
                batch=8,
                device=0,
                pretrained=pretrained_weights,
                name='smokerdetector_stage_2',
                project=SAVE_DIR,
                lr0=0.001,
                lrf=0.0001,
                mosaic=1.0)
    
    print("\n✅ Tuning complete!")
    print(f"📁 Results saved to: {os.path.join(SAVE_DIR, 'smoker_detector')}")
    print("📊 Use TensorBoard or check 'results.png' for loss & accuracy graphs.")
    print("🔥 Best model: best.pt (use this for detection)\n")

if __name__ == "__main__":
    train()