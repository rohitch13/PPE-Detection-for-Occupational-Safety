import os
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Set paths
dataset_path = "ppe_human_detection"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# YOLO class map
class_map = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",         # Background/No relevant object
    6: "person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots"
}


# Aggregate all label files
label_files = glob.glob(f"{labels_path}/**/*.txt", recursive=True)

# Data holders
class_counts = defaultdict(int)
bbox_dims = []
image_shapes = []
missing_labels = []

print(f"Processing {len(label_files)} label files...\n")

for label_file in label_files:
    with open(label_file, "r") as f:
        lines = f.readlines()

    if not lines:
        missing_labels.append(label_file)

    # Get corresponding image path
    rel_path = label_file.replace("labels", "images").replace(".txt", ".jpg")
    if not os.path.exists(rel_path):
        rel_path = rel_path.replace(".jpg", ".png")

    if not os.path.exists(rel_path):
        continue

    img = cv2.imread(rel_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    image_shapes.append((w, h))

    for line in lines:
        cls, *bbox = map(float, line.strip().split())
        class_counts[int(cls)] += 1

        x_center, y_center, bw, bh = bbox
        bbox_dims.append((bw, bh))

# Convert bbox dims to dataframe
bbox_df = pd.DataFrame(bbox_dims, columns=["width", "height"])

# Plot class distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=[class_map[k] for k in class_counts.keys()],
            y=list(class_counts.values()))
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Plot bounding box sizes
plt.figure(figsize=(6, 6))
sns.scatterplot(x=bbox_df["width"], y=bbox_df["height"], alpha=0.3)
plt.title("Normalized Bounding Box Width vs Height")
plt.xlabel("Width")
plt.ylabel("Height")
plt.grid(True)
plt.show()

# Image resolution distribution
df_shapes = pd.DataFrame(image_shapes, columns=["width", "height"])
plt.figure(figsize=(8, 4))
sns.histplot(df_shapes["width"], kde=True, label='Width')
sns.histplot(df_shapes["height"], kde=True, color="orange", label='Height')
plt.legend()
plt.title("Image Resolution Distribution")
plt.xlabel("Pixels")
plt.show()

# Missing label stats
print(f"\n🧪 Summary:")
print(f"Total images analyzed: {len(image_shapes)}")
print(f"Missing label files: {len(missing_labels)}")
if missing_labels:
    print(f"Examples: {missing_labels[:3]}")
