# Import necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# Set up the environment
# !pip install opencv-python tensorflow flask

# 1. Data Preparation
# Load the dataset (using a sample dataset for illustration)
# Let's use CIFAR-10 dataset for simplicity
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Visualize some images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(x_train[i])
    ax.axis('off')
plt.show()

# 2. Building the Model
# Load the VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
output = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Training the Model
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test), epochs=10)

# 4. Evaluating the Model
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Validation Accuracy: {accuracy*100:.2f}%')

# 5. Saving and Loading the Model
# Save the model
model.save('cifar10_vgg19_model.h5')

