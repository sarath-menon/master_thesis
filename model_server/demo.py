from models.florence2 import Florence2Model
from PIL import Image

def main():
    model = Florence2Model()
    image = Image.open("/Users/sarathmenon/Documents/master_thesis/datasets/resized_media/gameplay_images/hogwarts_legacy/0.jpg")

    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    text_input = 'sword.'
    results = model.run_inference(image, task_prompt, text_input=text_input)

    print(results)

if __name__ == "__main__":
    main()

