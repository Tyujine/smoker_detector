from ultralytics import YOLO
import os

DATASET_PATH = 'dataset/Smoker Detector-New/'
DATA_YAML = DATASET_PATH + 'data.yaml'
MODEL_NAME = 'yolo11n.pt'
SAVE_DIR = 'runs/train/'

def train():
    model = YOLO(MODEL_NAME)
    model.info()
    model.train(data=DATA_YAML,
                epochs=50,
                imgsz=640,
                batch=16,
                device=0,
                pretrained=True,
                name='smokerdetector_stage_1',
                project=SAVE_DIR,
                augment=True,
                patience=20)
    
    print("\n✅ Training complete!")
    print(f"📁 Results saved to: {os.path.join(SAVE_DIR, 'smoker_detector')}")
    print("📊 Use TensorBoard or check 'results.png' for loss & accuracy graphs.")
    print("🔥 Best model: best.pt (use this for detection)\n")

if __name__ == "__main__":
    train()