import os
import cv2
import numpy as np

# Define directories
RAW_DIR = "dataset/raw"
PROCESSED_DIR = "dataset/processed"
IMG_SIZE = (224, 224)  # Standard size for CNN input

def preprocess_and_save(image_path, save_path):
    """Load, preprocess, and save image."""
    image = cv2.imread(image_path)  # Load image
    if image is None:
        print(f"Error loading {image_path}")
        return

    image = cv2.resize(image, IMG_SIZE)  # Resize image
    image = image / 255.0  # Normalize (0-1)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save preprocessed image
    cv2.imwrite(save_path, (image * 255).astype(np.uint8))
    print(f"Saved: {save_path}")

def process_dataset():
    """Loop through raw dataset and preprocess all images."""
    for cancer_type in os.listdir(RAW_DIR):  # Loop through cancer types
        cancer_path = os.path.join(RAW_DIR, cancer_type)
        if not os.path.isdir(cancer_path):
            continue

        for category in ["Training", "Testing"]:  # Train & Test folders
            category_path = os.path.join(cancer_path, category)
            if not os.path.isdir(category_path):
                continue

            for label in ["Normal", "Tumor"]:  # Normal & Tumor
                label_path = os.path.join(category_path, label)
                if not os.path.isdir(label_path):
                    continue

                # Destination directory in processed folder
                save_dir = os.path.join(PROCESSED_DIR, cancer_type, category, label)

                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    save_path = os.path.join(save_dir, img_name)

                    preprocess_and_save(img_path, save_path)

if __name__ == "__main__":
    process_dataset()
    print("✅ Preprocessing complete! All images saved to processed folder.")
