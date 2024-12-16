from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests


print(torch.cuda.is_available())
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_name = "llava-hf/llama3-llava-next-8b-hf"
model_directory = f"{model_name.split('/')[1]}-ai"

try:
    model = LlavaNextProcessor.from_pretrained(model_directory, torch_dtype=torch.float16, device_map="auto")
    processor = LlavaNextProcessor.from_pretrained(model_directory, do_resize=False)
except:
    model = LlavaNextProcessor.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.save_pretrained(model_directory)
    processor = LlavaNextProcessor.from_pretrained(model_name, do_resize=False)
    processor.save_pretrained(model_directory)
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