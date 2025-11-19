import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# --------------------------------------------
# Custom Fog Function
# --------------------------------------------
def add_synthetic_fog(image):
    h, w = image.shape[:2]
    fog_density = np.random.uniform(0.2, 0.6)
    fog = np.full((h, w, 3), 255, dtype=np.uint8)
    blurred_fog = cv2.GaussianBlur(fog, (0, 0), sigmaX=np.random.uniform(30, 60))
    result = cv2.addWeighted(image, 1 - fog_density, blurred_fog, fog_density, 0)
    return result

# --------------------------------------------
# Augmentation Pipeline
# --------------------------------------------
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.6),

    A.MotionBlur(blur_limit=7, p=0.4),
    A.GaussianBlur(blur_limit=7, p=0.4),

    A.GaussNoise(noise_limit=(10, 50), p=0.4),
    A.ISONoise(color_shift=(0.01, 0.07), intensity=(0.1, 0.5), p=0.4),

    A.ImageCompression(quality_range=(40, 95), p=0.4),

    A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, alpha_coef=0.1, p=0.3),

    A.OneOf([
        A.ColorJitter(),
        A.ToGray(),
        A.ChannelShuffle()
    ], p=0.3),

    A.Lambda(image=lambda img, **kwargs: add_synthetic_fog(img) if np.random.rand() < 0.3 else img)
], 
bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

# --------------------------------------------
# Paths
# --------------------------------------------
input_img_dir = r"dataset\Smoker Detector-New\train\images"
input_lbl_dir = r"dataset\Smoker Detector-New\train\labels"

output_img_dir = r"dataset_augmented\train\images"
output_lbl_dir = r"dataset_augmented\train\labels"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# --------------------------------------------
# Process Images + Labels
# --------------------------------------------
for filename in tqdm(os.listdir(input_img_dir)):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(input_img_dir, filename)
    lbl_path = os.path.join(input_lbl_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not read {filename}")
        continue

    # Load labels
    if not os.path.exists(lbl_path):
        print(f"No label for {filename}")
        continue

    bboxes = []
    class_labels = []

    with open(lbl_path, "r") as f:
        for line in f.readlines():
            c, x, y, w, h = line.strip().split()
            bboxes.append([float(x), float(y), float(w), float(h)])
            class_labels.append(int(c))

    # Apply augmentation
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

    aug_img = augmented["image"]
    aug_boxes = augmented["bboxes"]
    aug_labels = augmented["class_labels"]

    # Save image
    cv2.imwrite(os.path.join(output_img_dir, filename), aug_img)

    # Save updated labels
    with open(os.path.join(output_lbl_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt")), "w") as f:
        for cls, (x, y, w, h) in zip(aug_labels, aug_boxes):
            f.write(f"{cls} {x} {y} {w} {h}\n")

print("YOLO image + label augmentation complete.")
