#%%import asyncio
from typing import List, Tuple
from PIL import Image
import numpy as np
from ..pipeline.core import Pipeline, PipelineState, PipelineStep, PipelineMode, PipelineModeSequence, PipelineModes
from ..common.data_structures import *
from .pipeline_wrapper import PipelineWrapper
from ..prompt_refinement.core import PromptRefiner, PromptMode
from ..image_processor.pointing import Pointing, PointingInput
from ..output_corrector.core import VerificationMode
from clicking_client import Client

class MolmoDirectPipelineWrapper(PipelineWrapper):
    def initialize_pipeline(self):
        # Initialize components
        client = Client(base_url=self.config['api']['local_url'], timeout=120)
        prompt_refiner = PromptRefiner(config=self.config)
        pointing_processor = Pointing(client, config=self.config)

        # Define pipeline modes
        self.pipeline_modes = PipelineModes({
            "pointing_input_mode": PointingInput,
            "pointing_mode": TaskType
        })

        # Create pipeline steps
        pipeline_steps = [
            PipelineStep(
                name="Get Clickpoints",
                function=pointing_processor.get_pointing_results,
                mode_keys=["pointing_mode", "pointing_input_mode"]
            ),
        ]

        # Create pipeline and add steps
        pipeline = Pipeline(config=self.config)
        for step in pipeline_steps:
            pipeline.add_step(step)

        return pipeline

    async def process_image(self, image: Image.Image, obj_name: str) -> List[Tuple[float, float]]:

        # Create a ClickingImage object
        clicking_image = ClickingImage(image=image, id=str(uuid.uuid4()), path="")
        clicking_image.predicted_objects = [ImageObject(name=obj_name, description=obj_name)]
        
        # Create initial state
        initial_state = PipelineState(images=[clicking_image])
        
        # Run the pipeline
        pipeline_mode_sequence = PipelineModeSequence.from_config(self.config, self.pipeline_modes)
        all_results = await self.pipeline.run_for_all_modes(
            initial_state=initial_state,
            pipeline_modes=pipeline_mode_sequence,
            start_from_step="Get Clickpoints",
            #stop_after_step="Get Clickpoints"
        )
        
        # Extract clickpoints from the results
        result = all_results.get_run_by_mode_name("direct_clickpoint")
        clickpoints = []
        for obj in result.images[0].predicted_objects:
            if obj.clickpoint:
                return obj.clickpoint
            else:
                print(f"Clickpoint detection failed")
