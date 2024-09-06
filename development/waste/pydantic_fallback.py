#%%
from pydantic import BaseModel, field_validator
from enum import Enum

class ObjectCategory(Enum):
    GAME_ASSET = "GAME_ASSET"
    NON_PLAYABLE_CHARACTER = "NON_PLAYABLE_CHARACTER"
    OTHER = "OTHER"

class MyModel(BaseModel):
    name: str
    age: int
    category: ObjectCategory

    # @field_validator('category', mode='before')
    # def validate_category(cls, value):
    #     try:
    #         return ObjectCategory(value)
    #     except ValueError:
    #         return ObjectCategory.OTHER  # Default fallback value

# Example usage
data = {"name": "John", "age": 30, "category": "NON_PLAYABLE_CHARACTER1"}

#%%

models = []
for category in ObjectCategory:
    try:
        model = MyModel(**data)
        models.append(model)
    except Exception as e:
        print(f"Error: {e}")
# %%
