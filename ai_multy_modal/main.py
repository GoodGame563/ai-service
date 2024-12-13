from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16, device_map="auto")

# prepare image and text prompt, using the appropriate prompt template
url = "image/5.webp"
image = Image.open(url)


conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Проанализируй текст на карточке товара и предложи конкретные советы по улучшению его читаемости. Например, предложи увеличить размер шрифта, изменить цвет шрифта или настроить интервал между строками. Убедись, что рекомендации включают точные значения (например, 'Увеличить размер шрифта до 16px' или 'Изменить цвет на #FFFFFF')."},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=200)

print(processor.decode(output[0], skip_special_tokens=True))
input()