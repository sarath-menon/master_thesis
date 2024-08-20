from litellm import completion
import os
import dotenv
import base64
import io

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class InstructionToLabel:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant and an expert videogame player."},
        ]

    def _get_prompt(self, text_prompt: str):
        prompt = f"""
        Analyze the videogame screenshot. For the action "{text_prompt}":
        1. Identify the game object to selected to execute the action. 
        2. Provide a brief 10-word reasoning.
        
        Return JSON: {{
            "class_label": "object to click",
            "reasoning": "10-word explanation"
        }}
        """
        return prompt
    

    def _get_text_response(self, prompt: str):
        prompt = self._get_prompt(prompt)
        self.add_message(prompt)
        response = completion(model=self.model, messages=self.messages)
        return response["choices"][0]["message"]["content"]
    
    def _get_image_response(self, base64_image: str, text_prompt: str):
        msg = {"role": "user", "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}

        self.messages.append(msg)
        response = completion(model=self.model, messages=self.messages)
        return response["choices"][0]["message"]["content"]
    
    def _pil_to_base64(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_label(self, screenshot: str, action: str):
        base64_image = self._pil_to_base64(screenshot)
        full_prompt = self._get_prompt(action)
        return self._get_image_response(base64_image, full_prompt)

    def show_messages(self):
        for message in self.messages:
            print(message)