import os
import cv2
import numpy as np
from tqdm import tqdm

# Set image size
IMAGE_SIZE = (224, 224)  # Resize images to 224x224

def preprocess_images(input_path, output_path):
    """
    Preprocess images by resizing and normalizing, then save to the output directory.
    """
    categories = ["Tumor", "Normal"]  # Categories in the dataset

    for category in categories:
        input_category_path = os.path.join(input_path, category)
        output_category_path = os.path.join(output_path, category)

        # Ensure output directory exists
        os.makedirs(output_category_path, exist_ok=True)

        print(f"Processing {category} images...")

        for img_name in tqdm(os.listdir(input_category_path), desc=f"Processing {category}"):
            img_path = os.path.join(input_category_path, img_name)

            try:
                # Read and preprocess the image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping {img_name}: Unable to read image.")
                    continue

                img = cv2.resize(img, IMAGE_SIZE)  # Resize
                img = img / 255.0  # Normalize

                # Save processed image
                output_img_path = os.path.join(output_category_path, img_name)
                cv2.imwrite(output_img_path, (img * 255).astype(np.uint8))  # Convert back to uint8

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

# Run preprocessing for both Training and Testing datasets
if __name__ == "__main__":
    preprocess_images(
        "D:/cancer detection/cancerDetection/dataset/raw/breast_cancer/Training",
        "D:/cancer detection/cancerDetection/dataset/processed/breast_cancer/Training"
    )

    preprocess_images(
        "D:/cancer detection/cancerDetection/dataset/raw/breast_cancer/Testing",
        "D:/cancer detection/cancerDetection/dataset/processed/breast_cancer/Testing"
    )

    print("✅ Breast cancer dataset preprocessing complete!")
