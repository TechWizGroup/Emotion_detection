# Nhập các thư viện cần thiết
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Kích hoạt mixed precision để tăng tốc độ trên GPU
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Kiểm tra GPU
print("Using GPU:", tf.config.list_physical_devices('GPU'))

# Đường dẫn dữ liệu và mô hình
data_path_train = '/kaggle/input/fer2013/train'
data_path_test = '/kaggle/input/fer2013/test'
model_path = '/kaggle/working/mo_hinh_nhan_dien_cam_xuc.h5'
best_model_path = '/kaggle/working/best_model.h5'

# Bật tiếp tục huấn luyện nếu cần
continue_training = True
batch_size = 128

# Layer augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Tạo dataset
def create_dataset(data_path, batch_size, shuffle=True):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(48, 48),
        batch_size=batch_size,
        shuffle=shuffle
    )
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.prefetch(tf.data.AUTOTUNE)

# Load dữ liệu train/test
train_dataset = create_dataset(data_path_train, batch_size, shuffle=True)
test_dataset = create_dataset(data_path_test, batch_size, shuffle=False)

# Xây dựng mô hình CNN (VGG-style + BN + Dropout)
model = Sequential([
    tf.keras.Input(shape=(48, 48, 1)),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax', dtype='float32')  # float32 vì softmax
])

# Tải model nếu đã có
if continue_training and os.path.exists(model_path):
    model.load_weights(model_path)
    print("Đã tải mô hình từ:", model_path)
else:
    print("Huấn luyện mô hình mới...")

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks thông minh
callbacks = [
    ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(patience=7, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    callbacks=callbacks
)

# Lưu mô hình
model.save(model_path)

# In kiến trúc
model.summary()

# --- VẼ BIỂU ĐỒ ACCURACY ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy mỗi Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()