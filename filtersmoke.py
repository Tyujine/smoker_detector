import os

# 🔧 Set your dataset path here (edit this line!)
DATASET_PATH = r"C:\Users\khunp\OneDrive\Documents\VSCode\ImageProject\dataset\Indoor Fire Smoke"  # e.g., "C:/Users/you/Desktop/fire_smoke_dataset"

# The dataset splits
splits = ['train', 'valid', 'test']

for split in splits:
    label_dir = os.path.join(DATASET_PATH, split, 'labels')
    image_dir = os.path.join(DATASET_PATH, split, 'images')
    print(f"\nProcessing {split} split...")

    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(label_dir, label_file)
        image_path = os.path.join(image_dir, label_file.replace('.txt', '.jpg'))

        # Read label content
        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls == 1:  # smoke
                parts[0] = '0'  # relabel smoke as class 0
                new_lines.append(' '.join(parts))

        if new_lines:
            # keep only smoke objects
            with open(label_path, 'w') as f:
                f.write('\n'.join(new_lines))
        else:
            # no smoke objects left → remove both label and image
            os.remove(label_path)
            if os.path.exists(image_path):
                os.remove(image_path)

print("\n✅ Done filtering. Only smoke images remain.")
