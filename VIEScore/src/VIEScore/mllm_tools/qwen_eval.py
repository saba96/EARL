
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVL():
    def __init__(self) -> None:
        print("Loading Qwen2.5-VL-72B model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct",
            # device_map="auto",
            device_map="cuda:1",
            load_in_4bit=True,
            torch_dtype=torch.bfloat16
        ).eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        print("Model loaded successfully!")

    def prepare_prompt(self, image_links: list = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]
        
        # Format the content list with proper image objects
        content = []
        for image in image_links:
            if isinstance(image, str):
                content.append({"type": "image", "image_url": image})
            else:
                content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text_prompt})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def get_parsed_output(self, inputs):
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generate_ids)
            ]
            generated_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return generated_text


if __name__ == "__main__":
    model = QwenVL()
    prompt = model.prepare_prompt(['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    res = model.get_parsed_output(prompt)
    print("result : \n", res) 