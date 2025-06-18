# Emotion Detection System

A real-time facial emotion recognition system that uses a deep learning model to detect seven emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) from webcam feed.

## Features

- Real-time facial emotion detection from webcam
- Color-coded emotion display
- Emotion updates every 2 seconds to reduce computational load
- Advanced face detection with multiple parameter configurations
- Visual status indicators

## Requirements

- Python 3.6+
- OpenCV
- TensorFlow 2.x
- NumPy

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the emotion recognition model (`best_model.h5`) and place it in the project directory
4. Run the application: `python main.py`

## Usage

- Press 'q' to exit the application
- The application will detect faces and display the emotion with a colored box
- Emotion updates happen every 2 seconds

## Project Structure

- `main.py`: The main application entry point
- `image_emotion_recognizer.py`: Contains the logic for image-based emotion recognition
- `emotion_recognizer.py`: Abstract base class for emotion recognizers
- `requirements.txt`: List of Python dependencies
- `best_model.h5`: Pre-trained emotion recognition model (not included in repo)

## Model Information

The emotion detection uses a CNN model trained on the FER2013 dataset, recognizing 7 basic emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral
