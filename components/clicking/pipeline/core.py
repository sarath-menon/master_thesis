from typing import Callable, List, Any, Dict, Tuple, TypedDict, get_origin, get_args
import inspect
import matplotlib.pyplot as plt
from PIL import Image

class Pipeline:
    def __init__(self):
        self.steps: List[Tuple[Callable, bool]] = []

    def add_step(self, func: Callable, verbose: bool = True):
        if self.steps and not self._are_types_compatible(self.steps[-1][0], func):
            last_step_func = self.steps[-1][0]
            last_step_output_type = inspect.signature(last_step_func).return_annotation.__name__
            next_step_input_type = inspect.signature(func).parameters[next(iter(inspect.signature(func).parameters))].annotation.__name__

            raise TypeError(f"Output type of {last_step_func.__name__} ({last_step_output_type}) is not compatible with input type of {func.__name__} ({next_step_input_type})")
        self.steps.append((func, verbose))

    def _are_types_compatible(self, prev_func: Callable, next_func: Callable) -> bool:
        prev_return_type = inspect.signature(prev_func).return_annotation
        next_param_types = [param.annotation for param in inspect.signature(next_func).parameters.values()]
        
        if not next_param_types:
            return True
        
        if prev_return_type == Any or next_param_types[0] == Any:
            return True
        
        # Handle TypedDict and other complex types
        if get_origin(prev_return_type) is not None:
            return self._check_complex_type_compatibility(prev_return_type, next_param_types[0])
        
        # For simple types, use isinstance check instead of issubclass
        return isinstance(prev_return_type, type(next_param_types[0]))

    def _check_complex_type_compatibility(self, type1, type2):
        origin1, origin2 = get_origin(type1), get_origin(type2)
        args1, args2 = get_args(type1), get_args(type2)
        
        if origin1 is TypedDict and origin2 is TypedDict:
            # For TypedDict, check if all keys in type2 exist in type1
            return all(key in type1.__annotations__ for key in type2.__annotations__)
        
        if origin1 is origin2:
            # For other complex types (List, Dict, etc.), check their arguments
            return all(self._are_types_compatible(arg1, arg2) for arg1, arg2 in zip(args1, args2))
        
        return False

    def run(self, initial_input: Any) -> Any:
        result = initial_input
        for step, verbose in self.steps:
            result = step(result)
            if verbose:
                self._log_step_result(step.__name__, result)
        return result

    def _log_step_result(self, step_name: str, result: Any):
        print(f"\n--- Step: {step_name} ---")
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, Image.Image):
                    print(f"{key}: <PIL.Image.Image object>")
                    plt.figure(figsize=(5, 5))
                    plt.imshow(value)
                    plt.axis('off')
                    plt.title(f"{step_name} - {key}")
                    plt.show()
                elif isinstance(value, list) and all(isinstance(item, Image.Image) for item in value):
                    print(f"{key}: <List of PIL.Image.Image objects>")
                    fig, axes = plt.subplots(1, len(value), figsize=(5*len(value), 5))
                    if len(value) == 1:
                        axes = [axes]
                    for i, (ax, img) in enumerate(zip(axes, value)):
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(f"{step_name} - {key}[{i}]")
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"{key}: {value}")
        else:
            print(result)

    def static_analysis(self):
        if not self.steps:
            raise ValueError("Pipeline has no steps.")
        
        for i in range(len(self.steps) - 1):
            current_step, _ = self.steps[i]
            next_step, _ = self.steps[i + 1]
            
            current_return_type = inspect.signature(current_step).return_annotation
            next_param_types = list(inspect.signature(next_step).parameters.values())
            
            if not next_param_types:
                continue
            
            next_param_type = next_param_types[0].annotation
            
            if current_return_type == Any or next_param_type == Any:
                continue
            
            if not issubclass(current_return_type, next_param_type):
                raise TypeError(f"Output type of {current_step.__name__} ({current_return_type}) "
                                f"is not compatible with input type of {next_step.__name__} ({next_param_type})")
        
        print("Static analysis complete. All types are compatible.")

