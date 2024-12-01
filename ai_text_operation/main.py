from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.models as models
import torch

# Путь к локальной модели
model_directory = "./qwen_model"
try:
# Загрузка модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory)
except:
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    local_directory = "./qwen_model"

    # Загрузка модели и токенизатора с Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Сохранение модели и токенизатора локально
    tokenizer.save_pretrained(local_directory)
    model.save_pretrained(local_directory)


# Перемещение модели на доступное устройство (CPU или GPU)

print(torch.cuda.is_available())
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

device = torch.device("cuda")
model = model.to(device)
 
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
       
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

prompt = f"""Проанализируй два товара и определите, схожи ли они. Используй название и описание каждого товара для оценки. Сравни следующие характеристики:
Название: Обрати внимание на ключевые слова в названиях. Если названия имеют схожие слова или смысл, это может указывать на схожесть.
Описание: Рассмотри детали в описаниях. Обрати внимание на:
Основные функции или особенности товаров.
Материалы или ингредиенты, если применимо.
Целевую аудиторию или назначение товаров.
На основе этого анализа ответь 'True', если товары схожи, и 'False', если они не схожи.
'"""
messages = [
    {"role": "system", "content": "Ты модель которая отвечает True or False."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])