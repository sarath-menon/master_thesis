#%%
import inspect
from typing import List, Callable, Any
from pydantic import BaseModel, ValidationError, create_model
import inspect
#%%
class Task:
    func: Callable
    input_type: BaseModel
    output_type: BaseModel

    def __init__(self, func: Callable):
        self.func = func
        sig = inspect.signature(func)
        self.input_type = create_model('InputModel',__config__=ArbitraryTypeConfig, **{
            k: (v.annotation, ...) for k, v in sig.parameters.items()
        })
        self.output_type = create_model('OutputModel',__config__=ArbitraryTypeConfig, **{
            "result": (sig.return_annotation, ...)
        })

def create_pipeline(tasks: List[Task]):
    def pipeline(input_data: Any):
        current_data = input_data
        for task in tasks:
            # Validate input type
            try:
                current_data = task.input_type.parse_obj(current_data)
            except ValidationError as e:
                raise ValueError(f"Input validation failed for {task.func.__name__}: {e}")

            # Execute the task
            result = task.func(current_data.dict())

            # Validate output type
            try:
                current_data = task.output_type.parse_obj({"result": result})
            except ValidationError as e:
                raise ValueError(f"Output validation failed for {task.func.__name__}: {e}")

        return current_data
    return pipeline

def verify_pipeline(tasks: List[Task]):
    for i in range(len(tasks) - 1):
        current_output = tasks[i].output_type.schema()['properties']
        next_input = tasks[i + 1].input_type.schema()['properties']
        if current_output != next_input:
            raise TypeError(f"Output of {tasks[i].func.__name__} does not match input of {tasks[i + 1].func.__name__}")

# Example usage
tasks = [Task(func=get_localization_prediction.sync), Task(func=get_segmentation_prediction.sync)]
pipeline_func = create_pipeline(tasks)
verify_pipeline(tasks)
# input_data = {"image": b'sample_image_data'}
# output = pipeline_func(input_data)

#%%
from clicking_client.api.default import  get_localization_prediction,get_segmentation_prediction
import inspect
from pydantic import BaseModel, create_model, ConfigDict

# Custom configuration to allow arbitrary types
class ArbitraryTypeConfig(ConfigDict):
    arbitrary_types_allowed = True

sig = inspect.signature(get_localization_prediction.sync)
print(sig)
input_type = create_model('InputModel',__config__=ArbitraryTypeConfig, **{
            k: (v.annotation, ...) for k, v in sig.parameters.items()
        })
output_type = create_model('OutputModel',__config__=ArbitraryTypeConfig, **{
    "result": (sig.return_annotation, ...)
})

print(input_type)
#%%

# Setup the pipeline
tasks = [
    Task(func=localization_task),
    Task(func=segmentation_task),
]

# Create the pipeline function
pipeline_func = create_pipeline(tasks)
input_data = {"image": b'sample_image_data'}
output = pipeline_func(input_data)
