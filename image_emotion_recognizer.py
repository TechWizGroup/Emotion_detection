import cv2
import numpy as np
from tensorflow.keras.models import load_model
from emotion_recognizer import EmotionRecognizer

class ImageEmotionRecognizer(EmotionRecognizer):
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recognize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Thử với các tham số khác nhau để tăng khả năng phát hiện khuôn mặt
        scale_factors = [1.1, 1.2, 1.3, 1.05]
        min_neighbors_options = [3, 4, 5]
        
        faces = []
        # Thử với các tham số khác nhau nếu không tìm thấy khuôn mặt
        for scale in scale_factors:
            for min_neighbors in min_neighbors_options:
                faces = self.face_cascade.detectMultiScale(gray, scale, min_neighbors)
                if len(faces) > 0:
                    break
            if len(faces) > 0:
                break
        
        # Nếu vẫn không tìm thấy, thử với cân bằng histogram để cải thiện độ tương phản
        if len(faces) == 0:
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            preds = self.model.predict(roi, verbose=0)[0]
            emotion = self.emotion_labels[np.argmax(preds)]
            results.append({'box': (x, y, w, h), 'emotion': emotion})
        return results
