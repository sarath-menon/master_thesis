from models import Florence2Model, GroundingDinoModel
from PIL import Image

def main():
    #model = Florence2Model()
    model = GroundingDinoModel()
    image = Image.open("/Users/sarathmenon/Documents/master_thesis/datasets/resized_media/gameplay_images/hogwarts_legacy/10.jpg")

    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    text_input = 'the books. the person.'
    results = model.run_inference(image, task_prompt, text_input=text_input)

    print(results)

if __name__ == "__main__":
    main()

