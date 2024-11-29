from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16, device_map="auto") 

# prepare image and text prompt, using the appropriate prompt template
url = "image/2.webp"
image = Image.open(url)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "Опиши все, что изображено на предоставленной картинке. Укажи объекты, их взаимное расположение, цвета, размеры, действия (если изображено движение), а также стиль изображения. Ответ должен быть исключительно на русском языке, максимально детализированным и развернутым."},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=800)

print(processor.decode(output[0], skip_special_tokens=True))
