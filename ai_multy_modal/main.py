from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,BitsAndBytesConfig
import torch
from PIL import Image
import requests

config = BitsAndBytesConfig(
    quantization_config={"load_in_8bit": True}
)

print(torch.cuda.is_available())
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_name = "llava-hf/llama3-llava-next-8b-hf"
model_directory = f"{model_name.split('/')[1]}-ai"

# try:
#     model = LlavaNextProcessor.from_pretrained(model_directory, torch_dtype=torch.float16, device_map="auto")
#     processor = LlavaNextProcessor.from_pretrained(model_directory, do_resize=False)
# except:
#     model = LlavaNextProcessor.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
#     model.save_pretrained(model_directory)
#     processor = LlavaNextProcessor.from_pretrained(model_name, do_resize=False)
#     processor.save_pretrained(model_directory)

model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", use_flash_attention_2=True)
processor = LlavaNextProcessor.from_pretrained(model_name, do_resize=False)
url = "image/5.webp"


model = torch.compile(model)
def analyze_photo(url):
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

  output = model.generate(**inputs, max_new_tokens=800,  temperature=0.7, do_sample=True).cpu()

  print(processor.decode(output[0], skip_special_tokens=True))

def analyze_photo_by_text(url):
  result = []
  image = Image.open(url)
  conversation = [
      {
        "role": "user",
        "content": [
            {"type": "text", "text": "Оцени текст на изображении: насколько легко его прочитать? Укажи, виден ли текст четко, не сливается ли он с фоном."},
            {"type": "image"},
          ],
      },
  ]
  prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

  inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

  output = model.generate(**inputs, max_new_tokens=800,  temperature=0.7, do_sample=True).cpu()

  result.append(str(processor.decode(output[0], skip_special_tokens=True)).split(".assistant"))
  conversation = [
      {
        "role": "user",
        "content": [
            {"type": "text", "text": "Определи, подходит ли стиль шрифта данному изображению. Уместен ли он для карточки товара? Отнеси стиль шрифта к категории (например, строгий, декоративный, минималистичный)."},
            {"type": "image"},
          ],
      },
  ]
  prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

  inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

  output = model.generate(**inputs, max_new_tokens=800,  temperature=0.7, do_sample=True).cpu()

  result.append(str(processor.decode(output[0], skip_special_tokens=True)).split(".assistant"))
  print(result)

analyze_photo_by_text(url)