

# Dataset Path
train_dir = r"D:\cancer detection\cancerDetection\dataset\processed\breast_cancer\Testing"
val_dir = r"D:\cancer detection\cancerDetection\dataset\processed\breast_cancer\Testing"


# Image Preprocessing
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_val = ImageDataGenerator(rescale=1./255)

train_generator = datagen_train.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)
val_generator = datagen_val.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)

# Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary Classification (Benign or Malignant)
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the Model
model.save("D:/cancer detection/cancerDetection/models/train_cnn_breast.h5")
print("✅ Breast Cancer CNN trained and saved.")
