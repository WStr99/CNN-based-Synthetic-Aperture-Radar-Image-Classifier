import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

# === Settings ===
xml_dir = r"C:\Users\wills\Desktop\VOC2007\Annotations"
img_dir = r"C:\Users\wills\Desktop\VOC2007\JPEGImages"
output_dir = r"C:\Users\wills\Desktop\VOC2007\yolo_dataset"
splits = ['train', 'val', 'test']
split_ratio = [0.8, 0.1, 0.1]  # train, val, test

# === Create YOLO folders ===
for split in splits:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# === Helper: Convert VOC box to YOLO format ===
def voc_to_yolo(box, img_width, img_height):
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return [x_center, y_center, width, height]

# === Load and filter annotations ===
def load_voc_annotations(xml_dir, img_dir):
    samples = []
    for file in os.listdir(xml_dir):
        if not file.endswith(".xml"): continue
        xml_path = os.path.join(xml_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        if '_flip' in filename: continue  # ignore flipped images
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path): continue

        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        boxes = []
        labels = []
        for obj in root.findall('object'):
            class_id = int(obj.find('name').text)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append(voc_to_yolo([xmin, ymin, xmax, ymax], width, height))
            labels.append(class_id)

        samples.append({
            'image': img_path,
            'filename': filename,
            'boxes': boxes,
            'labels': labels
        })

    return samples

# === Split and save ===
def save_yolo_dataset(samples):
    train, val_test = train_test_split(samples, test_size=1-split_ratio[0], random_state=42)
    val, test = train_test_split(val_test, test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), random_state=42)
    split_data = {'train': train, 'val': val, 'test': test}

    for split in splits:
        for sample in split_data[split]:
            # Copy image
            dst_img = os.path.join(output_dir, 'images', split, sample['filename'])
            shutil.copy(sample['image'], dst_img)

            # Save label file
            label_path = os.path.splitext(sample['filename'])[0] + '.txt'
            dst_label = os.path.join(output_dir, 'labels', split, label_path)

            with open(dst_label, 'w') as f:
                for box, cls in zip(sample['boxes'], sample['labels']):
                    line = f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n"
                    f.write(line)

# === Run everything ===
samples = load_voc_annotations(xml_dir, img_dir)
save_yolo_dataset(samples)
print("YOLO dataset created at:", output_dir)