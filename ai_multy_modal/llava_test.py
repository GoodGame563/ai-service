from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"

model_directory = f"{model_name.split('/')[1]}"


try:
    model = LlavaNextForConditionalGeneration.from_pretrained(model_directory, torch_dtype="auto", device_map="auto")
    processor = LlavaNextProcessor.from_pretrained(model_directory)
except:
    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    processor = LlavaNextProcessor.from_pretrained(model_name)
    processor.save_pretrained(model_directory)
    model.save_pretrained(model_directory)

model.to("cuda:0")

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
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

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=800)

print(processor.decode(output[0], skip_special_tokens=True))
