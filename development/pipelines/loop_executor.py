#%%
import yaml
import time
import logging
from PIL import Image
import asyncio
from clicking.pipelines.molmo_direct import MolmoDirectPipelineWrapper
from clicking.image_processor.visualization import show_clickpoint_crosshair

class LoopExecutor:
    def __init__(self, config_path: str):
        self.setup_logger()
        
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
            
        self.pipeline = MolmoDirectPipelineWrapper(self.config)
        
    def setup_logger(self):
        self.logger = logging.getLogger('LoopExecutor')
        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        self.logger.setLevel(logging.INFO)
        
        # Create console handler and set formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)

    async def execute_sequence_async(
        self, 
        sequence_file: str, 
        get_image_func,
        delay: float = 0.0, 
        show_image: bool = False
    ):
        """
        Executes a sequence of text inputs from a YAML file with specified delay between each.
        
        Args:
            sequence_file (str): Path to YAML file containing sequence of text inputs
            get_image_func (callable): Function that returns a PIL Image for each iteration
            delay (float): Delay in seconds between executing each input
            show_image (bool): Whether to display the image with clickpoint
            
        Yields:
            tuple: (text_input, clickpoint, step_number, total_steps)
        """
        self.logger.info(f"Loading sequence from {sequence_file}")
        
        with open(sequence_file, 'r') as f:
            sequence_data = yaml.safe_load(f)
        
        instructions = sequence_data.get('instructions', [])
        if not instructions or not isinstance(instructions, list):
            self.logger.error("Invalid sequence file format. Expected 'instructions' key with list of text inputs.")
            return
            
        for i, text_input in enumerate(instructions):
            img = get_image_func()
            if not isinstance(img, Image.Image):
                self.logger.error(f"get_image_func returned invalid type: {type(img)}. Expected PIL Image.")
                continue

            # Check if instruction starts with "delay"
            if isinstance(text_input, str) and text_input.startswith('delay'):
                try:
                    # Split instruction and get delay value
                    _, delay_seconds = text_input.split()
                    delay_seconds = int(delay_seconds)
                    self.logger.info(f"Delaying for {delay_seconds} seconds")
                    time.sleep(delay_seconds)
                except (ValueError, IndexError):
                    self.logger.error(f"Invalid delay instruction format: {text_input}")
                
                continue
                
            clickpoint = await self.pipeline.process_image(img, text_input)
            
            self.logger.info(f"Step {i+1}/{len(instructions)}: Object: {text_input}, Clickpoint: ({clickpoint.x}, {clickpoint.y})")
            
            if show_image:
                show_clickpoint_crosshair(img, clickpoint)
                
            yield img, clickpoint,text_input  # Now yield both image and clickpoint
            time.sleep(delay)

    def execute_sequence(
        self, 
        sequence_file: str, 
        get_image_func,
        delay: float = 0.0, 
        show_image: bool = False
    ):
        loop = asyncio.get_event_loop()
        async_gen = self.execute_sequence_async(sequence_file, get_image_func, delay, show_image)
        while True:
            try:
                img, clickpoint, text_input = loop.run_until_complete(anext(async_gen))  
                yield img, clickpoint, text_input
            except StopAsyncIteration:
                break

# #%%
# import nest_asyncio
# nest_asyncio.apply()

# if __name__ == "__main__":
#     config_path = "./development/pipelines/game_object_config.yml"
#     executor = LoopExecutor(config_path)
    
#     sequence_file = "./development/prompt_sequences/unpacking.yaml"
    
#     # Example function that returns a static image
#     def get_test_image():
#         return Image.open("./datasets/resized_media/gameplay_images/unpacking/6.jpg")
    
#     for img, clickpoint in executor.execute_sequence(
#         sequence_file, 
#         get_image_func=get_test_image,
#         show_image=True
#     ):
#         print(f"Clickpoint: x={clickpoint.x}, y={clickpoint.y}")
# # %%
