from fastapi.testclient import TestClient
from clicking.api import core
from clicking.vision_model.data_structures import TaskType, SetModelReq, PredictionReq, AutoAnnotationReq
import io

client = TestClient(core.app)

def test_get_models():
    response = client.get("/vision_model/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert all("name" in model and "variants" in model and "tasks" in model for model in data["models"])

    # Extract and print supported task types for each model
    for model in data["models"]:
        print(f"Model: {model['name']}")
        print(f"Supported task types: {model['tasks']}")
        print("---")

    return data["models"]  # Return the models data for further use if needed

def test_get_model():
    # First, set a model
    set_model_req = SetModelReq(task=TaskType.LOCALIZATION_WITH_TEXT_GROUNDED, name="florence2", variant="florence-2-base")
    set_response = client.post("/vision_model/model", json=set_model_req.dict())
    if set_response.status_code != 200:
        print(f"Set model failed with status {set_response.status_code}")
        print(f"Response content: {set_response.content}")
    assert set_response.status_code == 200

    # Then, get the model
    get_response = client.get("/vision_model/model", params={"task": TaskType.LOCALIZATION_WITH_TEXT_GROUNDED.value})
    assert get_response.status_code == 200
    data = get_response.json()
    assert "name" in data
    assert "variant" in data
    assert data["name"] == "florence2"
    assert data["variant"] == "florence-2-base"

# def test_set_model():
#     req = SetModelReq(task=TaskType.LOCALIZATION_WITH_TEXT_GROUNDED, name="florence2", variant="florence-2-base")
#     response = client.post("/vision_model/model", json=req.dict())
#     if response.status_code != 200:
#         print(f"Set model failed with status {response.status_code}")
#         print(f"Response content: {response.content}")
#     assert response.status_code == 200
#     assert "message" in response.json()
#     assert "status_code" in response.json()

def test_prediction():
    # Create a mock image file
    image_file = io.BytesIO(b"fake image content")
    image_file.name = "test_image.jpg"

    # Set the model first
    set_model_req = SetModelReq(task=TaskType.LOCALIZATION_WITH_TEXT_GROUNDED, name="florence2", variant="florence-2-base")
    set_response = client.post("/vision_model/model", json=set_model_req.dict())
    assert set_response.status_code == 200

    # Prepare the prediction request
    files = {
        "image": ("test_image.jpg", image_file, "image/jpeg"),
    }
    form_data = {
        "input_text": "sample text",
    }
    params = {
        "task": TaskType.LOCALIZATION_WITH_TEXT_GROUNDED.value
    }

    response = client.post("/vision_model/prediction", files=files, data=form_data, params=params)
    print(f"Prediction response: {response.content}")  # Add this line for debugging
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_auto_annotation():
    # Create a mock image file
    image_file = io.BytesIO(b"fake image content")
    image_file.name = "test_image.jpg"

    # Set the model first
    set_model_req = SetModelReq(task=TaskType.SEGMENTATION_AUTO_ANNOTATION, name="sam2", variant="sam2_hiera_tiny")
    set_response = client.post("/vision_model/model", json=set_model_req.dict())
    assert set_response.status_code == 200

    # Prepare the auto annotation request
    files = {
        "image": ("test_image.jpg", image_file, "image/jpeg"),
    }
    params = {
        "task": TaskType.SEGMENTATION_AUTO_ANNOTATION.value
    }

    response = client.post("/vision_model/auto_annotation", files=files, params=params)
    print(f"Auto annotation response: {response.content}")  # Add this line for debugging
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_prediction_missing_task():
    # Create a mock image file
    image_file = io.BytesIO(b"fake image content")
    image_file.name = "test_image.jpg"

    req = {
        "image": ("test_image.jpg", image_file, "image/jpeg"),
        "params": '{"confidence": 0.5, "iou": 0.5}'
    }
    response = client.post("/vision_model/prediction", files=req)
    assert response.status_code == 400
    assert "detail" in response.json()

def test_auto_annotation_missing_task():
    # Create a mock image file
    image_file = io.BytesIO(b"fake image content")
    image_file.name = "test_image.jpg"

    req = {
        "image": ("test_image.jpg", image_file, "image/jpeg"),
        "params": '{"confidence": 0.5, "iou": 0.5}'
    }
    response = client.post("/vision_model/auto_annotation", files=req)
    assert response.status_code == 400
    assert "detail" in response.json()