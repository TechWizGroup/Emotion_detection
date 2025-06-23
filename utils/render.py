import cv2

def render_frame(frame):
    """
    Render a single frame using OpenCV.
    
    Args:
        frame: The frame to render, typically a NumPy array representing an image.
    """
    window_name = 'Facial Emotion Recognition'
    cv2.imshow(window_name, frame)

    # Check if window was closed (X pressed)
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()
        return False

    # Check if 'q' was pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False

    return True

def add_label(frame, text, position=(50, 50), font_scale=1, color=(0, 0, 0), background_color = (255, 255, 255), thickness=1):
    """
    Add text to a frame.
    
    Args:
        frame: The frame to which the text will be added.
        text: The text to add to the frame.
        position: The position (x, y) where the text will be placed.
        font_scale: Scale factor for the font size.
        color: Color of the text in BGR format.
        thickness: Thickness of the text.
    """
    x, y = position
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x - 5, y + 5), (x + w + 5, y - h - 5), background_color, thickness = -1)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)

def add_bounding_box(frame, box, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on the frame.
    
    Args:
        frame: The frame on which to draw the bounding box.
        box: A tuple (x, y, w, h) representing the bounding box.
        color: Color of the bounding box in BGR format.
        thickness: Thickness of the bounding box lines.
    """
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

def render_results(frame, position, result):
    """
    Render results on the frame.
    
    Args:
        frame: The frame to render results on.
        results: A list of tuples containing bounding boxes and labels.
    """
    # Định nghĩa màu sắc cho từng loại cảm xúcAdd commentMore actions
    emotion_colors = {
        'Angry': (0, 0, 200),      # Đỏ đậm
        'Disgust': (0, 140, 255),  # Cam
        'Fear': (0, 0, 255),       # Đỏ
        'Happy': (0, 255, 0),      # Xanh lá
        'Sad': (255, 0, 0),        # Xanh dương
        'Surprise': (255, 255, 0), # Vàng
        'Neutral': (150, 150, 150) # Xám
    }
    color = emotion_colors.get(result, (255, 255, 255))  # Mặc định là trắng nếu không tìm thấy cảm xúc
    x, y, w, h = position
    add_bounding_box(frame, (x, y, w, h), color)
    add_label(frame, result, position=(x, y-10))

    # Vẽ legend cho các loại cảm xúc
    legend_x = 10
    legend_y = 60
    for i, (emotion, color) in enumerate(emotion_colors.items()):
        cv2.putText(frame, f"{emotion}", 
                    (legend_x, legend_y + i*25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, 
                        (legend_x + 100, legend_y + i*25 - 15), 
                        (legend_x + 130, legend_y + i*25 + 5), 
                        color, -1)

    return render_frame(frame)