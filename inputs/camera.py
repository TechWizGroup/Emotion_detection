import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(self.camera_index)

    def start(self):
        if not self.capture.isOpened():
            raise Exception("Could not open video device")

    def read_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            raise Exception("Failed to read frame from camera")
        return frame

    def stop(self):
        self.capture.release()
        cv2.destroyAllWindows()