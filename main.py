from inputs import camera
from detector import Detector
import utils.render as render
from utils import image_utils

def facial_recognition():
    """
    Hàm chính để nhận diện khuôn mặt.
    """
    # Khởi tạo camera và bộ phát hiện khuôn mặt
    cam = camera.Camera()
    det = Detector(facial_mode=True, default_mode="facial")
    cam.start()
    while True:
        frame = cam.read_frame()

        face = image_utils.detect_faces(frame)
        if face is not None:
            x, y, w, h = face
            img = frame[y:y+h, x:x+w]
            result = det.predict(img, None)
            render.add_bounding_box(frame, (x, y, w, h), color=(0, 255, 0), thickness=2)
            render.add_label(frame, result, position=(x, y-10))

        if not render.render_frame(frame):
            break
    cam.stop()

if __name__ == "__main__":

    facial_recognition()