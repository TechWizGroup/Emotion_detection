import tensorflow as tf
import cv2

LABLES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def image_preprocess(image, target_size=(48, 48)):
    """
    Preprocesses the input image for model prediction.
    
    :param image: Input image to be preprocessed.
    :param target_size: Target size for the image.
    :return: Preprocessed image ready for model input.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert BGR to RGB
    image = image.astype('float') / 255.0
    image = tf.expand_dims(image, axis=-1)  # thêm trục channels vào cuối
    if image.shape[:2] != target_size:
        image = tf.image.resize(image, target_size)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postpredict(result):
    """
    Post-processes the model prediction result.
    
    :param result: Raw prediction result from the model.
    :return: Processed prediction result.
    """
    if result is None or len(result) == 0:
        raise ValueError("Prediction failed, no result returned.")
    
    label_index = tf.argmax(result[0]).numpy()
    label = LABLES.get(label_index, "Unknown")
    
    return label

def detect_faces(image):
    """
    Detects faces in the input image using OpenCV's Haar Cascade classifier.
    
    :param image: Input image in which to detect faces.
    :return: List of bounding boxes for detected faces.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda x: x[2] * x[3])  # Get the largest face by area