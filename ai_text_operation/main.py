from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.models as models
import torch
from pydantic import BaseModel


class SEO_word(BaseModel):
    word: str
    need_for: int
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
    prompt = f"В данном тексте есть грамматические ошибки {text}"
    messages = [
        {"role": "system", "content": """Вы — эксперт по языку и грамматике, ваша задача — анализировать текст и выявлять в нём ошибки. Это могут быть грамматические, пунктуационные, орфографические, стилистические или логические ошибки. Не учитывай слова из другого языка в тексте как ошибку. На основе этого анализа ответь если есть ошибки напиши False а если их нет то True'"""},
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
        max_new_tokens=6, 
        temperature=0.1
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if (len(str(final).split(" ")) > 1):
        print (final)
    return str(final) == "True"

def generate_new_text(text: str, seo_words: list[SEO_word]):
    if len(seo_words) == 0:
        prompt = f"{text}"
        messages = [
            {"role": "system", "content": """Прочитай текст, проанализируй его, и переформулируй так, чтобы он:
                    Сохранил оригинальный смысл и стиль описания товара.
                    Выглядел профессионально и грамотно, без орфографических или грамматических ошибок.
                    Подчеркнул уникальные преимущества и продающие характеристики товара, делая его более привлекательным для покупателей.
                    Не используй иностранные языки кроме названия товара.
                Пример:
                    Оригинал: 'Уникальный крем для лица на натуральной основе, который увлажняет кожу и делает её гладкой.'
                    Переформулированное: 'Натуральный крем для лица, который обеспечивает интенсивное увлажнение, придавая коже гладкость и сияние.'
                Вот текст для переформулирования:"""},
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
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print (final)
    

text = """Базовая футболка с усиленным воротом для мужчин – сочетание стиля и комфорта на каждый день. Глубокий черный цвет подчеркивает мужественность и добавляет образу харизмы, а V-образный вырез выгодно выделяет футболку среди других. Комфортная посадка дарит уют и свободу движений в любой ситуации.

Однотонная мужская футболка с V-вырезом выполнена из нежного 100% хлопка – трикотажной ткани кулирная гладь. Она отлично выдерживает многочисленные стирки, сохраняя форму и цвет. Дышащая мягкая ткань делает футболку идеальной для летней, весенней и осенней погоды. Короткие рукава и прямой пошив не сковывают движений и позволяют сохранять привычный ритм жизни. Аккуратные швы не ощущаются в течение дня, а усиленный ворот комфортно прилегает к телу.

Классическая мужская футболка отлично впишется в деловой образ для офиса и учебы, а также в уличный и smart casual (смарт кэжуал) стили. Черная футболка из трикотажа подойдет на лето, теплую весну и осень. Хлопковая футболка идеальна для отдыха дома, поездок и путешествий.

Рост модели 172 см, его параметры: объем груди 113 см, объем талии 82 см, объем бедер 100 см. На нем футболка размера 50."""

# print(len(text.split()))
# t, f = (0, 0)
# for _ in range(50):
#     if text_write_rigth(text): t += 1
#     else: f += 1

# print(f"True: {t}, False: {f}")

generate_new_text(text, [])