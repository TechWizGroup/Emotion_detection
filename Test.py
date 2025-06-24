# Kích thước đầu ra của từng lớp:


# Lớp	                    Kích thước đầu ra	        Ghi chú
# Input	                    [batch_size, 48, 48, 1]	    Hình ảnh grayscale 48x48.

# Conv2D_1 (64 filters)	    [batch_size, 48, 48, 64]	Padding='same' giữ kích thước 48x48, 64 kênh.
# BatchNormalization_1	    [batch_size, 48, 48, 64]	Không thay đổi kích thước. Chuẩn hóa từng kênh: (x - mean) / sqrt(var + epsilon) * gamma + beta.
# Conv2D_2 (64 filters)	    [batch_size, 48, 48, 64]	Padding='same', vẫn 64 kênh.
# BatchNormalization_2	    [batch_size, 48, 48, 64]	Không thay đổi kích thước. Ví dụ: Nếu đầu vào là [0.2, 0.4, 0.6], sau BN sẽ được chuẩn hóa về trung bình 0, phương sai 1 (theo batch).
# MaxPooling2D_1	        [batch_size, 24, 24, 64]	Pooling 2x2 giảm kích thước từ 48x48 xuống 24x24. Lấy giá trị lớn nhất trong mỗi vùng 2x2.
# Dropout_1	                [batch_size, 24, 24, 64]	Không thay đổi kích thước. Ngẫu nhiên đặt một tỉ lệ (ví dụ 25%) giá trị thành 0 trong quá trình train, ví dụ: [0.5, 0.8, 0.3] -> [0.5, 0, 0.3] nếu dropout 33%.

# Conv2D_3 (128 filters)	[batch_size, 24, 24, 128]	Padding='same', tăng kênh lên 128.
# BatchNormalization_3	    [batch_size, 24, 24, 128]	Không thay đổi kích thước. Chuẩn hóa từng kênh.
# Conv2D_4 (128 filters)	[batch_size, 24, 24, 128]	Padding='same', vẫn 128 kênh.
# BatchNormalization_4	    [batch_size, 24, 24, 128]	Không thay đổi kích thước.
# MaxPooling2D_2	        [batch_size, 12, 12, 128]	Pooling 2x2 giảm kích thước từ 24x24 xuống 12x12. Ví dụ: vùng [[1,2],[3,4]] -> 4.
# Dropout_2	                [batch_size, 12, 12, 128]	Không thay đổi kích thước. Ngẫu nhiên đặt giá trị thành 0.

# Conv2D_5 (256 filters)	[batch_size, 12, 12, 256]	Padding='same', tăng kênh lên 256.
# BatchNormalization_5	    [batch_size, 12, 12, 256]	Không thay đổi kích thước.
# Conv2D_6 (256 filters)	[batch_size, 12, 12, 256]	Padding='same', vẫn 256 kênh.
# BatchNormalization_6	    [batch_size, 12, 12, 256]	Không thay đổi kích thước.
# MaxPooling2D_3	        [batch_size, 6, 6, 256]	    Pooling 2x2 giảm kích thước từ 12x12 xuống 6x6.
# Dropout_3	                [batch_size, 6, 6, 256]	    Không thay đổi kích thước.

# Flatten	                [batch_size, 9216]	        6x6x256 = 9216 (vector 1D).
# Dense_1 (512 units)	    [batch_size, 512]	        Giảm kích thước xuống 512.
# BatchNormalization_7	    [batch_size, 512]	        Không thay đổi kích thước.
# Dropout_4	                [batch_size, 512]	        Không thay đổi kích thước.
# Dense_2 (7 units)	        [batch_size, 7]	            Đầu ra cuối: xác suất cho 7 cảm xúc.


# ==============================================================
# Ví dụ ảnh đầu vào (grayscale, không có kênh màu)
# Kích thước: 4x4
# x = [
#     [1, 2, 3, 4],
#     [2, 3, 4, 5],
#     [3, 4, 5, 6],
#     [4, 5, 6, 7]
# ]
# Bước 1: Tính mean và variance toàn bộ ảnh (16 pixel)
# Tổng: (1+2+...+7)*2 - 1 - 7 = 64 → mean = 64 / 16 = 4.0
# variance = trung bình (x - mean)^2
#           = [(1-4)^2 + (2-4)^2 + ... + (7-4)^2] / 16 = 3.5

# Bước 2: Áp dụng công thức chuẩn hóa:
# x_norm = (x - mean) / sqrt(var + epsilon)
# ví dụ: pixel 1 → (1 - 4) / sqrt(3.5 + 1e-5) ≈ -1.603

# Kết quả là ma trận đầu ra có giá trị mean~0, std~1

# ==============================================================
# Ví dụ tensor đầu vào: 4x4x4 (4 kênh feature map sau conv layer)
# Với mỗi pixel (i,j), có 4 giá trị (4 kênh)

# Tưởng tượng ta có 4 "bức ảnh" 4x4:
# channel_1 = [
#     [1, 2, 3, 4],
#     [2, 3, 4, 5],
#     [3, 4, 5, 6],
#     [4, 5, 6, 7]
# ]
# channel_2 = [
#     [10, 9, 8, 7],
#     [9, 8, 7, 6],
#     [8, 7, 6, 5],
#     [7, 6, 5, 4]
# ]
# channel_3 = [
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1]
# ]
# channel_4 = [
#     [0, 1, 0, 1],
#     [1, 0, 1, 0],
#     [0, 1, 0, 1],
#     [1, 0, 1, 0]
# ]

# BatchNorm sẽ tính toán riêng biệt cho từng kênh:
# Với mỗi channel:
#   - mean_channel_i = trung bình toàn bộ 16 giá trị
#   - var_channel_i = phương sai của 16 giá trị
#   - rồi chuẩn hóa: (x - mean_i) / sqrt(var_i + epsilon)

# Ví dụ:
#   Channel 1 → mean = 4.0, var ≈ 3.5
#   Channel 2 → mean = 7.0, var ≈ 3.5
#   Channel 3 → mean = 1.0, var = 0.0 → chuẩn hóa xong toàn bộ là 0
#   Channel 4 → mean = 0.5, var = 0.25

# Kết quả: mỗi channel đầu ra được chuẩn hóa độc lập → giữ tính phân biệt giữa các đặc trưng

# ==============================================================

# - Dropout (rate=0.5): Đầu vào là ma trận 8x8, ví dụ:
#   [[0.5,0.8,0.3,...], ...]
#   -> Một nửa số phần tử ngẫu nhiên bị đặt thành 0, ví dụ:
#   [[0.5,0,0.3,0,0,0.7,0,0.2], ...]

# - MaxPooling2D (2x2): Vùng 2x2 bất kỳ trong ma trận 8x8, ví dụ:
#   [[1, 2],
#    [3, 4]]
#   -> 4 (lấy max)
# =========================








import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
import os

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

# Path to the test image
img_path = r"C:/Users/USER/Desktop/PythonTest/EmotionDetection/Test.jpg"

# Load and preprocess the image
img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize

# Display original image
plt.figure(figsize=(5, 5))
plt.imshow(img_array[0, :, :, 0], cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.savefig(os.path.join(results_dir, 'original_image.png'))
plt.show()

# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Apply data augmentation
augmented_img = data_augmentation(img_array, training=True)
plt.figure(figsize=(5, 5))
plt.imshow(augmented_img[0, :, :, 0], cmap='gray')
plt.title('Augmented Image')
plt.axis('off')
plt.savefig(os.path.join(results_dir, 'augmented_image.png'))
plt.show()

# Replace the sequential model with a functional API model
inputs = tf.keras.Input(shape=(48, 48, 1))

# Block 1
x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv1')(inputs)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
x = Dropout(0.25)(x)

# Block 2
x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
x = Dropout(0.25)(x)

# Block 3
x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)
x = Dropout(0.25)(x)

# Classification block
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(7, activation='softmax', dtype='float32')(x)

# Create the visualization model
visualization_model = Model(inputs=inputs, outputs=outputs, name='emotion_model')

# Define layer names to visualize
layers_to_visualize = [
    'block1_conv1', 'block1_conv2', 'block1_pool',
    'block2_conv1', 'block2_conv2', 'block2_pool', 
    'block3_conv1', 'block3_conv2', 'block3_pool'
]

# Dictionary for custom layer titles
layer_titles = {
    'block1_conv1': 'Block 1 - Conv 1',
    'block1_conv2': 'Block 1 - Conv 2',
    'block1_pool': 'Block 1 - Pool',
    'block2_conv1': 'Block 2 - Conv 1',
    'block2_conv2': 'Block 2 - Conv 2',
    'block2_pool': 'Block 2 - Pool',
    'block3_conv1': 'Block 3 - Conv 1',
    'block3_conv2': 'Block 3 - Conv 2',
    'block3_pool': 'Block 3 - Pool'
}

# Function to visualize activations of a specific layer
def visualize_layer_activations(model, layer_name, input_img, display_title):
    # Get the layer
    layer = model.get_layer(layer_name)
    
    # Create a new model that outputs the target layer's output
    layer_model = Model(inputs=model.input, outputs=layer.output)
    
    # Get activations
    activations = layer_model.predict(input_img, verbose=0)
    
    # Normalize the activations for better visualization
    activations = (activations - activations.min()) / (activations.max() - activations.min() + 1e-10)
    
    # Get total number of channels
    n_channels = activations.shape[3]
    
    # Determine grid size based on number of channels
    if n_channels <= 16:
        grid_size = (4, 4)
    elif n_channels <= 36:
        grid_size = (6, 6)
    elif n_channels <= 64:
        grid_size = (8, 8)
    elif n_channels <= 100:
        grid_size = (10, 10)
    elif n_channels <= 144:
        grid_size = (12, 12)
    else:
        # Cap at 16x16 grid (256 maps) for very large layers
        grid_size = (16, 16)
        n_channels = min(256, n_channels)
    
    # Calculate figure size (proportional to grid size)
    fig_size = (grid_size[1]*3, grid_size[0]*3)
    
    # Create a figure to display the activations
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=fig_size)
    axes = axes.flatten()
    
    # Plot the activations
    for i in range(grid_size[0] * grid_size[1]):
        if i < n_channels:
            axes[i].imshow(activations[0, :, :, i], cmap='viridis')
            axes[i].set_title(f'Channel {i+1}')
        axes[i].axis('off')
    
    plt.suptitle(f'{display_title} Activations ({n_channels} channels)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the figure
    fig.savefig(os.path.join(results_dir, f'{layer_name}_activations.png'))
    plt.show()

# Visualize intermediate outputs for each layer in our list
for layer_name in layers_to_visualize:
    display_title = layer_titles.get(layer_name, layer_name)
    print(f"Visualizing activations for layer: {layer_name}")
    visualize_layer_activations(visualization_model, layer_name, augmented_img, display_title)

# Function to visualize the sequential transformation of the image
def visualize_feature_maps_progression(model, input_img):
    # Create a new output directory for the progression
    progression_dir = os.path.join(results_dir, 'progression')
    os.makedirs(progression_dir, exist_ok=True)
    
    # Visualize each layer in order
    for i, layer_name in enumerate(layers_to_visualize):
        # Create a model for this layer
        layer = model.get_layer(layer_name)
        temp_model = Model(inputs=model.input, outputs=layer.output)
        
        # Get output for this layer
        feature_maps = temp_model.predict(input_img, verbose=0)
        
        # Visualize a subset of feature maps for each layer
        n_features = min(4, feature_maps.shape[3])
        
        plt.figure(figsize=(12, 3))
        for j in range(n_features):
            plt.subplot(1, n_features, j+1)
            plt.imshow(feature_maps[0, :, :, j], cmap='viridis')
            plt.title(f'Feature {j+1}')
            plt.axis('off')
        
        plt.suptitle(f'Layer: {layer_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(progression_dir, f'{i:02d}_{layer_name}_progression.png'))
        plt.show()

# Visualize the progression of feature maps through the network
print("Visualizing image transformation through the network...")
visualize_feature_maps_progression(visualization_model, img_array)

print(f"Visualization complete! All images saved to '{results_dir}' folder.")