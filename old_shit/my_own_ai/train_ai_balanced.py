import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
import pickle

# 1. Load and preprocess the data
data_dir = 'onlyfaces'
batch_size = 32
img_size = (160, 160)
augmentation_factor = 5

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% of the data for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load all data without splitting to determine the minimum number of images per class
all_data_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Determine the minimum number of images per class
class_counts = np.sum(all_data_gen.labels, axis=0)
min_class_count = int(np.min(class_counts))

# Custom generator to yield equal number of images from each class
# Precompute balanced indices for each class
def get_balanced_indices(generator, min_class_count):
    # Directly use generator.labels as class indices
    class_indices = generator.labels
    
    # Fetch min_class_count indices for each class
    balanced_indices = []
    for cls in range(generator.num_classes):
        cls_indices = np.where(class_indices == cls)[0]
        np.random.shuffle(cls_indices)
        balanced_indices.extend(cls_indices[:min_class_count].tolist())
    
    np.random.shuffle(balanced_indices)
    return balanced_indices

# Custom generator using precomputed balanced indices
def balanced_generator(generator, balanced_indices):
    while True:
        for i in range(0, len(balanced_indices), generator.batch_size):
            batch_indices = balanced_indices[i:i+generator.batch_size]
            X = np.array([generator[i][0] for i in batch_indices])
            y = np.array([generator[i][1] for i in batch_indices])
            yield X, y






# Use the custom generator for training
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Compute balanced indices
balanced_indices = get_balanced_indices(train_gen, min_class_count)

# Use the custom generator for training
balanced_train_gen = balanced_generator(train_gen, balanced_indices)


# Save class labels
with open('class_labels.pkl', 'wb') as f:
    pickle.dump(train_gen.class_indices, f)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 2. Define the model architecture
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# 3. Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10
history = model.fit(balanced_train_gen, validation_data=val_gen, epochs=epochs)

# 4. Evaluate the model (optional)
loss, accuracy = model.evaluate(val_gen)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 5. Save the model
model.save('face_recognition_model.h5')
