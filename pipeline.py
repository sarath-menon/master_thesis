from typing import Callable, List, Any, Dict
import inspect

class Pipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, func: Callable):
        if self.steps and not self._are_types_compatible(self.steps[-1], func):
            raise TypeError(f"Output type of {self.steps[-1].__name__} is not compatible with input type of {func.__name__}")
        self.steps.append(func)

    def _are_types_compatible(self, prev_func: Callable, next_func: Callable) -> bool:
        prev_return_type = inspect.signature(prev_func).return_annotation
        next_param_types = [param.annotation for param in inspect.signature(next_func).parameters.values()]
        
        if not next_param_types:
            return True
        
        if prev_return_type == Any or next_param_types[0] == Any:
            return True
        
        return issubclass(prev_return_type, next_param_types[0])

    def run(self, initial_input: Any) -> Any:
        result = initial_input
        for step in self.steps:
            result = step(result)
        return result

def pipeline_step(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__annotations__ = func.__annotations__
    return wrapper