import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def GPU_set():
    """
    Thiết lập môi trường GPU cho TensorFlow
    """
    # Kích hoạt mixed precision để tăng tốc độ trên GPU
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Kiểm tra GPU
    print("Using GPU:", tf.config.list_physical_devices('GPU'))

class FacialModel:
    def __init__(self, pre_train=True, model_path='pretrain/facial_model.weights.h5'):

        # Xây dựng mô hình CNN (VGG-style + BN + Dropout)
        self.model = Sequential([
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

        if pre_train:
            # Nạp mô hình đã huấn luyện nếu không cần huấn luyện lại
            self.model.load_weights(model_path)
            print("Mô hình đã được nạp từ:", model_path)
        

    def fit(self, train_dataset, test_dataset, best_model_path='best_model.h5'):

        # Biên dịch
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Thiết lập các callback
        callbacks = [
            ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_accuracy', mode='max'),
            EarlyStopping(patience=7, monitor='val_loss', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]

        self.history = self.model.fit(
            train_dataset,
            epochs=100,
            validation_data=test_dataset,
            callbacks=callbacks
        )


    def save(self, model_path):
        self.model.save_weight(model_path)
        print("Mô hình đã được lưu tại:", model_path)
    
    def evaluate(self, test_dataset):
        results = self.model.evaluate(test_dataset)
        print("Đánh giá mô hình:", results)
        return results
    
    def predict(self, image):
        """
        Dự đoán cảm xúc từ ảnh đầu vào.
        
        :param image: Ảnh đầu vào đã được tiền xử lý.
        :return: Kết quả dự đoán từ mô hình.
        """
        prediction = self.model.predict(image, verbose=0)
        return prediction
    
    def summary(self):
        self.model.summary()

class FacialDataLoader:
    
    def __init__(self, batch_size=128,
                data_path_train = 'data/fer2013/train',
                data_path_test = 'data/fer2013/test'):
        self.batch_size = batch_size
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test

        # Layer augmentation
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1)
        ])

    # Tạo dataset
    def create_dataset(self, data_path, batch_size, shuffle=True):
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
            dataset = dataset.map(lambda x, y: (self.data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=1000)
        return dataset.prefetch(tf.data.AUTOTUNE)

    def load_data(self):
        # Tạo dataset train và test
        train_dataset = self.create_dataset(self.data_path_train, self.batch_size, shuffle=True)
        test_dataset = self.create_dataset(self.data_path_test, self.batch_size, shuffle=False)
        return train_dataset, test_dataset


