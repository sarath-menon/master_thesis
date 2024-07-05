from robyn import Robyn
import os
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import io
import base64
from models.florence2 import Florence2Model

def load_model(model_id):
    from transformers.dynamic_module_utils import get_imports
    from unittest.mock import patch

    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("/modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

def run_example( image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    early_stopping=False,
    do_sample=False,
    num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

app = Robyn(__file__)
model_ = Florence2Model()
model, processor = load_model('microsoft/Florence-2-base-ft')

@app.get("/")
async def h(request):
    return "Hello, world1!"

@app.get("/detection")
async def detection(req):
    req_json = req.json()
    base64_image = req_json['image']
    text_input = req_json['text_input']
    task_prompt = req_json['task_prompt']

    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))

    
    results = run_example(image, task_prompt, text_input=text_input)

    response = {
        "bboxes": results[task_prompt]['bboxes'],
        "labels": results[task_prompt]['labels']
    }
    return response

app.start(port=8082)