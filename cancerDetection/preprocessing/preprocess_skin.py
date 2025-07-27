import os
import pandas as pd
import cv2
from tqdm import tqdm

# ✅ Define Paths
RAW_DIR = "cancerDetection/dataset/raw/skin_cancer"
PROCESSED_DIR = "cancerDetection/dataset/processed/skin_cancer"

# 🔍 Metadata CSV File (Located in Testing folder)
METADATA_PATH = os.path.join(RAW_DIR, "Testing", "HAM10000_metadata.csv")
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"❌ Metadata file not found at: {METADATA_PATH}")

# ✅ Load Metadata CSV
df = pd.read_csv(METADATA_PATH)

# 🔹 Define Tumor & Normal Categories
tumor_labels = ["mel", "bcc", "akiec"]  # Malignant labels (Tumor)
df["label"] = df["dx"].apply(lambda x: "Tumor" if x in tumor_labels else "Normal")

# ✅ Create Processed Data Folders
for category in ["Training", "Testing"]:
    for label in ["Tumor", "Normal"]:
        os.makedirs(os.path.join(PROCESSED_DIR, category, label), exist_ok=True)

# 🔍 Find Image Directories (Part 1 and Part 2)
image_dirs = [
    os.path.join(RAW_DIR, "Training", "HAM10000_images_part_1"),
    os.path.join(RAW_DIR, "Training", "HAM10000_images_part_2"),
    os.path.join(RAW_DIR, "Testing", "HAM10000_images_part_1"),
    os.path.join(RAW_DIR, "Testing", "HAM10000_images_part_2"),
]

# ✅ Preprocessing Function
def preprocess_and_save_images(category):
    """Preprocess and save images into Normal & Tumor folders."""
    print(f"📂 Processing {category} Dataset...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row["image_id"]  # Get image ID (without extension)
        label = row["label"]  # Tumor or Normal
        
        # 🔍 Search for the image in available directories
        img_path = None
        for img_dir in image_dirs:
            for ext in [".jpg", ".png"]:  # Check for both JPG & PNG formats
                potential_path = os.path.join(img_dir, img_name + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            if img_path:  # If found, break
                break
        
        if img_path is None:
            print(f"⚠️ Warning: Image {img_name} not found in any directories!")
            continue

        # 🖼️ Load & Preprocess Image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping corrupt image: {img_name}")
            continue

        img = cv2.resize(img, (224, 224))  # Resize to 224x224 for CNN
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # 📂 Save Image in Correct Folder
        save_path = os.path.join(PROCESSED_DIR, category, label, img_name + ".jpg")
        cv2.imwrite(save_path, img)

    print(f"✅ {category} Dataset Preprocessing Complete!")

# 🔥 Run Preprocessing for Training & Testing Datasets
preprocess_and_save_images("Training")
preprocess_and_save_images("Testing")

print("🎉 Skin Cancer Dataset Preprocessing Completed Successfully!")
