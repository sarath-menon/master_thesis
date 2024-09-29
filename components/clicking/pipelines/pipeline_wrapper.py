import asyncio
from typing import List, Tuple
from PIL import Image
import numpy as np

class PipelineWrapper:
    def __init__(self, config):
        self.config = config
        self.pipeline_modes = None
        self.pipeline = self.initialize_pipeline()

    def initialize_pipeline(self):
        # Initialize the pipeline based on the configuration
        # This method should be implemented for each specific pipeline
        raise NotImplementedError

    async def process_image(self, image: Image.Image, text_input: str) -> List[Tuple[float, float]]:
        # Process the image using the pipeline
        # This method should be implemented for each specific pipeline
        raise NotImplementedError