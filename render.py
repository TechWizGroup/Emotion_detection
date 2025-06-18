import cv2

def render_frame(frame):
    """
    Render a single frame using OpenCV.
    
    Args:
        frame: The frame to render, typically a NumPy array representing an image.
    """
    cv2.imshow('Rendered Frame', frame)
    
    # Wait for a key press and close the window if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
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