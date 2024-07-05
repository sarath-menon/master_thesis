from abc import ABC, abstractmethod

class BaseClickingModel(ABC):
    def __init__(self, model_id):
        self.model, self.processor = self.load_model(model_id)

    @abstractmethod
    def load_model(self, model_id):
        pass

    @abstractmethod
    def run_inference(self, image, task_prompt, text_input=None):
        pass