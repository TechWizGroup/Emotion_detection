import cv2
import time
from image_emotion_recognizer import ImageEmotionRecognizer

def main():
    recognizer = ImageEmotionRecognizer('best_model.h5')
    cap = cv2.VideoCapture(0)
    
    # Đảm bảo webcam được mở thành công
    if not cap.isOpened():
        print("Không thể mở webcam")
        return
    
    print("Nhấn 'q' để thoát")
    
    # Định nghĩa màu sắc cho từng loại cảm xúc
    emotion_colors = {
        'Angry': (0, 0, 200),      # Đỏ đậm
        'Disgust': (0, 140, 255),  # Cam
        'Fear': (0, 0, 255),       # Đỏ
        'Happy': (0, 255, 0),      # Xanh lá
        'Sad': (255, 0, 0),        # Xanh dương
        'Surprise': (255, 255, 0), # Vàng
        'Neutral': (150, 150, 150) # Xám
    }
    
    # Biến để theo dõi thời gian
    last_process_time = 0
    process_interval = 2.0  # Khoảng thời gian giữa các lần xử lý (2 giây)
    results = []  # Lưu kết quả nhận diện gần nhất
    processing = False  # Trạng thái đang xử lý
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ webcam")
            break

        # Tạo bản sao của frame để vẽ lên
        display_frame = frame.copy()
        
        current_time = time.time()
        time_since_last = current_time - last_process_time
        
        # Chỉ xử lý cảm xúc mỗi 2 giây
        if time_since_last >= process_interval and not processing:
            processing = True
            # Nhận diện cảm xúc
            results = recognizer.recognize(frame)
            last_process_time = current_time
            processing = False
        
        # Hiển thị trạng thái xử lý và thời gian chờ
        if processing:
            status_text = "Processing..."
            status_color = (0, 255, 255)  # Vàng
        else:
            remaining = max(0, process_interval - time_since_last)
            status_text = f"Next update in: {remaining:.1f}s"
            status_color = (0, 255, 0) if remaining < 0.5 else (0, 200, 200)
            
        cv2.putText(display_frame, status_text, 
                   (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Hiển thị số lượng khuôn mặt được phát hiện
        cv2.putText(display_frame, f"Detected: {len(results)} face(s)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Vẽ kết quả lên frame
        for res in results:
            x, y, w, h = res['box']
            emotion = res['emotion']
            
            # Lấy màu tương ứng với cảm xúc
            color = emotion_colors.get(emotion, (255, 0, 0))
                
            # Vẽ hộp với độ dày 2 pixel
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Tạo nền cho text
            text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(display_frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
            
            # Viết cảm xúc
            cv2.putText(display_frame, emotion, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Vẽ legend cho các loại cảm xúc
        legend_x = 10
        legend_y = 60
        for i, (emotion, color) in enumerate(emotion_colors.items()):
            cv2.putText(display_frame, f"{emotion}", 
                      (legend_x, legend_y + i*25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(display_frame, 
                         (legend_x + 100, legend_y + i*25 - 15), 
                         (legend_x + 130, legend_y + i*25 + 5), 
                         color, -1)

        # Hiển thị frame
        cv2.imshow('Emotion Detection', display_frame)
        
        # Thoát nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
