from ultralytics import YOLO
import os

DATASET_PATH = 'dataset/Smoker Detector-New/'
DATA_YAML = DATASET_PATH + 'data.yaml'
MODEL_NAME = 'yolov8l.pt'
SAVE_DIR = 'runs/train/'

def train():
    model = YOLO(MODEL_NAME)
    model.load(SAVE_DIR + 'smoker_detector_New_2/weights/best.pt')
    model.info()
    model.train(data=DATA_YAML,
                epochs=50,
                imgsz=512,
                batch=16,
                device=0,
                lr0=0.001,
                workers=4,
                cache='disk',
                name='smoker_detector_Better_1',
                project=SAVE_DIR,
                augment=True,
                exist_ok=True,
                resume=False)
    
    print("\n✅ Training complete!")
    print(f"📁 Results saved to: {os.path.join(SAVE_DIR, 'smoker_detector')}")
    print("📊 Use TensorBoard or check 'results.png' for loss & accuracy graphs.")
    print("🔥 Best model: best.pt (use this for detection)\n")

if __name__ == "__main__":
    train()