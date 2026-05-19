from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, img_directory, split_images, split_classes):
        self.img_directory = img_directory
        self.split_images = split_images
        self.split_classes = split_classes
    
    @abstractmethod
    def get_text_prompt(self):
        """
        Abstract method for retrieving the prompt used during inference.
        """
        pass

    @abstractmethod
    def infer(self, img, text, **kwargs):
        """
        Abstract method for performing inference on a single image with a given text prompt.
        """
        pass
