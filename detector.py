import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.image_utils import *
import os

class Detector:
    def __init__(self, model_path="pretrain/", facial_mode=True, speech_mode=False, context_mode=False, is_post_process=False , default_mode="facial"):
        
        """ Initializes the Detector with specified model paths and modes.
        :param model_path: Path to the pre-trained models.
        :param facial_mode: Boolean indicating if the facial model should be loaded.
        :param speech_mode: Boolean indicating if the speech model should be loaded.
        :param context_mode: Boolean indicating if the context model should be loaded.
        :param is_post_process: Boolean indicating if post-processing should be applied to the predictions.
        :param default_mode: Default mode for post-processing.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

        self.facial_model = load_model(model_path+"facial_model.h5") if facial_mode else None
        self.speech_model = load_model(model_path+"speech_model.h5") if speech_mode else None
        self.context_model = load_model(model_path+"context_model.h5") if context_mode else None
        self.facial_mode = facial_mode
        self.speech_mode = speech_mode
        self.context_mode = context_mode
        self.is_post_process = is_post_process
        self.default_mode = default_mode

    def predict(self, image, speech):
        """
        Predicts the class of the input image.

        :param image: Input image to be classified.
        :param speech: Input speech data to be classified.
        :return: Predicted class label.
        """
        if self.facial_mode:
            facial_label = self.facial_predict(image) if self.facial_model else None
        if self.speech_mode:
            pass
        if self.context_mode:
            pass
        if not self.facial_mode and not self.speech_mode and not self.context_mode:
            raise ValueError("No model is loaded for prediction.")
        
        # Combine results from different models if needed
        predicted_results = {
            "facial": facial_label,
            # "speech": speech_label,
            # "context": context_label
        }
        result = self.post_process(predicted_results)

        return result

    def facial_predict(self, image):
        """
        Predicts the class of the input image using the facial model.

        :param image: Input image to be classified.
        :return: Predicted class label.
        """
        if self.facial_model is None:
            raise ValueError("Facial model is not loaded.")
        
        image = image_preprocess(image)
        if image is None:
            raise ValueError("Image preprocessing failed.")
        
        result = self.facial_model.predict(image, verbose=0)

        label = postpredict(result)
        if label is None or label == "Unknown":
            raise ValueError("Prediction failed, no valid label returned.")
        
        return label
    
    def post_process(self, result):
        """
        Post-processes the model prediction result.

        :param result: Raw prediction result from the model.
        :return: Processed prediction result.
        """
        
        if not self.is_post_process:
            return result[self.default_mode]

        return None
