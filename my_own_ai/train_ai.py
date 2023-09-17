import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
import pickle
from sklearn.utils.class_weight import compute_class_weight

# 1. Load and preprocess the data
# data_dir = 'onlyfaces'
data_dir = 'I:/.shortcut-targets-by-id/1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs/Åsalena/jpg_only_face_facerec'
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

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)


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

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weights = dict(enumerate(class_weights))

# Use class weights in model training
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, class_weight=class_weights)


# 4. Evaluate the model (optional)
loss, accuracy = model.evaluate(val_gen)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 5. Save the model
model.save('face_recognition_model.h5')
