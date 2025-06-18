from abc import ABC, abstractmethod

class EmotionRecognizer(ABC):
    @abstractmethod
    def recognize(self, input_data):
        pass
