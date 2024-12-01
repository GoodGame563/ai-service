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

print(torch.cuda.is_available())
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

device = torch.device("cuda")
model = model.to(device)

def text_write_rigth(text:str) -> bool:
    prompt = f"В данном тексте допущены ли грамматические ошибки {text}"
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
        max_new_tokens=5
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print (final)
    return str(final) == "True"


text = """Базовая футболка с усиленным воротом для мужчин – сочетание стиля и комфорта на каждый день. Глубокий черный цвет подчеркивает мужественность и добавляет образу харизмы, а V-образный вырез выгодно выделяет футболку среди других. Комфортная посадка дарит уют и свободу движений в любой ситуации.

Однотонная мужская футболка с V-вырезом выполнена из нежного 100% хлопка – трикотажной ткани кулирная гладь. Она отлично выдерживает многочисленные стирки, сохраняя форму и цвет. Дышащая мягкая ткань делает футболку идеальной для летней, весенней и осенней погоды. Короткие рукава и прямой пошив не сковывают движений и позволяют сохранять привычный ритм жизни. Аккуратные швы не ощущаются в течение дня, а усиленный ворот комфортно прилегает к телу.

Классическая мужская футболка отлично впишется в деловой образ для офиса и учебы, а также в уличный и smart casual (смарт кэжуал) стили. Черная футболка из трикотажа подойдет на лето, теплую весну и осень. Хлопковая футболка идеальна для отдыха дома, поездок и путешествий.

Рост модели 172 см, его параметры: объем груди 113 см, объем талии 82 см, объем бедер 100 см. На нем футболка размера 50."""

f, t = (0, 0)
for _ in range(50):
    if text_write_rigth(text): t+=1
    else: f+=1

print(f"false - {f} true - {t}")