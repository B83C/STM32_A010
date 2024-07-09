import numpy as np
import cv2
import random
import os


def read_binary_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    images = []
    labels = []
    entry_size = 1 + 100 * 100
    num_entries = len(data) // entry_size
    for i in range(num_entries):
        label = data[i * entry_size]
        image = np.frombuffer(
            data[(i * entry_size + 1) : (i + 1) * entry_size], dtype=np.uint8
        )
        image = image.reshape(100, 100)
        labels.append(label)
        images.append(image)
    return np.array(images), np.array(labels)


def write_binary_file(file_path, images, labels):
    with open(file_path, "wb") as f:
        for label, image in zip(labels, images):
            f.write(bytes([label]))
            f.write(image.tobytes())


def augment_image(image):
    # Random rotation
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((50, 50), angle, 1)
    rotated = cv2.warpAffine(image, M, (100, 100))

    # Random scaling
    scale = random.uniform(0.9, 1.1)
    scaled = cv2.resize(
        rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    )
    if scaled.shape[0] < 100 or scaled.shape[1] < 100:
        pad_y = (100 - scaled.shape[0]) // 2
        pad_x = (100 - scaled.shape[1]) // 2
        scaled = cv2.copyMakeBorder(
            scaled,
            pad_y,
            100 - scaled.shape[0] - pad_y,
            pad_x,
            100 - scaled.shape[1] - pad_x,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    else:
        start_y = (scaled.shape[0] - 100) // 2
        start_x = (scaled.shape[1] - 100) // 2
        scaled = scaled[start_y : start_y + 100, start_x : start_x + 100]

    # Random flipping
    if random.choice([True, False]):
        scaled = cv2.flip(scaled, 1)

    return scaled


def augment_dataset(images, labels, augmentations_per_image=5):
    augmented_images = []
    augmented_labels = []
    for image, label in zip(images, labels):
        for _ in range(augmentations_per_image):
            augmented_image = augment_image(image)
            print(f"{augmented_image.shape}")
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)


input_file_path = "training_data.bin"
output_file_path = "training_data_augmented.bin"

# Read original dataset
images, labels = read_binary_file(input_file_path)

# Augment dataset
augmented_images, augmented_labels = augment_dataset(images, labels)

# Combine original and augmented datasets
combined_images = np.concatenate((images, augmented_images), axis=0)
combined_labels = np.concatenate((labels, augmented_labels), axis=0)

# Write combined dataset to output binary file
write_binary_file(output_file_path, combined_images, combined_labels)
