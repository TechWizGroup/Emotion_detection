from models.facial.model import *

def train_facial_model():
    """
    Huấn luyện mô hình nhận diện cảm xúc khuôn mặt
    """
    # Thiết lập GPU
    GPU_set()

    # Khởi tạo mô hình
    facial_model = FacialModel()
    # FacialDataLoader
    facial_data_loader = FacialDataLoader()
    # Load data
    train_dataset, test_dataset = facial_data_loader.load_data()
    # Huấn luyện mô hình
    facial_model.fit(train_dataset, test_dataset)
    # Lưu mô hình
    facial_model.save('models/facial/model.weights.h5')
    print("Mô hình đã được huấn luyện và lưu thành công.")
    
def evaluate_facial_model():
    """
    Đánh giá mô hình nhận diện cảm xúc khuôn mặt
    """
    # Thiết lập GPU
    GPU_set()

    # Khởi tạo mô hình
    facial_model = FacialModel()
    # FacialDataLoader
    facial_data_loader = FacialDataLoader()
    # Load data
    _, test_dataset = facial_data_loader.load_data()
    # Đánh giá mô hình
    results = facial_model.evaluate(test_dataset)
    print("Kết quả đánh giá mô hình:", results)