from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.models as models
import torch
from pydantic import BaseModel
import json

class description_words(BaseModel):
    description: str
    words: list[str]

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

def generate_new_text_without_seo_words(text: str)->str:
    prompt = f"{text}"
    messages = [
        {"role": "system", "content": """
            Прочитай текст, проанализируй его, и переформулируй так, чтобы он:
                Сохранил оригинальный смысл и стиль описания товара.
                Выглядел профессионально и грамотно, без орфографических или грамматических ошибок.
                Подчеркнул уникальные преимущества и продающие характеристики товара, делая его более привлекательным для покупателей.
                Не используй иностранные языки кроме названия товара и бренда.
            Пример:
                Оригинал: 'Уникальный крем для лица на натуральной основе, который увлажняет кожу и делает её гладкой.'
                Переформулированное: 'Натуральный крем для лица, который обеспечивает интенсивное увлажнение, придавая коже гладкость и сияние.'
            Вот информация для переформулирования:"""},
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
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return final

def generate_new_text_with_seo_words_v1(main_element:description_words, elements:list[description_words]) -> str:
    prompt = f"Прочитай описание нашего товара и список ключевых слов:\nНаше описание: {main_element.description}\nНаши ключевые слова: {main_element.words}\nПрочитай описание и ключевые слова конкурентов (без упоминания их брендов):"
    for element in  elements:
        prompt += f"\nОписание конкурента: {element.description}\nКлючевые слова конкурента: {element.words}"
    
    messages = [
        {"role": "system", "content": """
    Ты — эксперт по созданию описаний товаров для маркетплейсов. Твоя задача — сравнить описание и ключевые слова нашего товара с описаниями и ключевыми словами конкурентов и улучшить его.

    Выдели и запомни ключевые слова и фразы из описаний конкурентов, которых нет в нашем описании, но которые могут сделать наше предложение более привлекательным.

    Перепиши описание нашего товара так, чтобы оно:
    Соответствовало формату карточки товара на маркетплейсе.
    Включало недостающие ключевые слова и фразы, сохраняя уникальность текста.
    Подчеркивало основные преимущества и уникальные характеристики товара.
    Привлекало внимание и мотивировало к покупке, избегая сложных терминов и профессионального жаргона.
    Используй следующий стиль ответа:
    'Ваше описание товара уже достаточно привлекательное и информативное, однако есть несколько моментов, которые можно улучшить:
        {здесь перечисли моменты которые можно улучшить в тексте}
    Описание товара:
        {Пишешь новое описание товара}'          """},
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
        temperature=0.8,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return final


def generate_new_text_with_seo_words_v2(main_element:description_words, elements:list[description_words]) -> str:
    prompt = f"Прочитай описание нашего товара и список ключевых слов:\nНаше описание: {main_element.description}\nНаши ключевые слова: {main_element.words}\nПрочитай описание и ключевые слова конкурентов (без упоминания их брендов):"
    for element in  elements:
        prompt += f"\nОписание конкурента: {element.description}\nКлючевые слова конкурента: {element.words}"
    
    messages = [
        {"role": "system", "content": """
    Ты — эксперт по созданию описаний товаров для маркетплейсов. Твоя задача — сравнить описание и ключевые слова нашего товара с описаниями и ключевыми словами конкурентов и улучшить его.

    Выдели и запомни ключевые слова и фразы из описаний конкурентов, которых нет в нашем описании, но которые могут сделать наше предложение более привлекательным.

    Перепиши описание нашего товара так, чтобы оно:
    Соответствовало формату карточки товара на маркетплейсе.
    Включало недостающие ключевые слова и фразы, сохраняя уникальность текста.
    Подчеркивало основные преимущества и уникальные характеристики товара.
    Привлекало внимание и мотивировало к покупке, избегая сложных терминов и профессионального жаргона.
    Используй следующий стиль ответа:
    'Ваше описание товара уже достаточно привлекательное и информативное, однако есть несколько моментов, которые можно улучшить:
        {здесь перечисли моменты которые можно улучшить в тексте}
    Описание товара:
        {Пишешь новое описание товара}'          """},
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
        temperature=1.1,
        length_penalty =0.8,
        num_beams = 3,
        top_p=0.9,
        early_stopping = True,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return final

# text = "Профессиональный спрей кондиционер для волос ULAB станет незаменимым помощником в ежедневном уходе за волосами. Этот продукт обеспечивает надежную термозащиту, разглаживание, увлажнение и восстановление. \n\nСпрей для волос обеспечивает защиту при укладке, благодаря чему волосы остаются гладкими и мягкими. Термозащитный эффект спрея предотвращает повреждение волос, сохраняя их силу и блеск. Этот спрей также защищает волосы от УФ-лучей.\n\nСредство солнцезащитное способствует легкому расчесыванию и укладке, благодаря чему волосы становятся более послушными. Спрей придает объем у корней для прикорневого объема, делая волосы густыми. Если ваши волосы склонны к пушистости или секущимся кончикам, этот спрей поможет разгладить их и придать волосам ухоженный вид. Эффект ламинирования, который дает спрей, делает волосы шелковистыми и блестящими.\n\nПарфюмированный спрей идеально подходит для использования на поврежденных, тонких и вьющихся волосах, обеспечивая им дополнительное питание и увлажнение, и защиту от солнца. Кератин в составе спрея способствует восстановлению структуры волос.\n\nЭтот кератиновый разглаживающий спрей для волос подходит для ухода за кожей головы, предотвращая сухость и шелушение. Он обеспечивает комплексный уход, питая волосы от корней до кончиков. \n\nЖенский спрей станет отличным выбором для тех, кто стремится к салонному уходу, к гладкости и блеску волос. Станет прекрасным подарком для женщин и девушек.\n\nСухие, поврежденные, пушистые или тонкие - многофункциональный спрей юлаб 10 в 1 обеспечит уход и восстановление. \n\nПрофессиональный несмываемый термоспрей для волос, особенно для окрашенных, обеспечивает защиту и эластичность, предотвращая электризацию и антистатик. Он идеален после утюжка и фена, сохраняя эффект кератинового выпрямления. Этот бьюти продукт незаменим для здорового и красивого вида волос."
# generate_new_text_without_seo_words(text)
# print(len(text.split()))
# t, f = (0, 0)
# for _ in range(50):
#     if text_write_rigth(text): t += 1
#     else: f += 1

# print(f"True: {t}, False: {f}")

main_element = description_words(description="",words=[])
elements = []

with open("response.json", "r", encoding='utf-8') as write_file:
    t = list(json.load(write_file))
    words = []
    
    first_element = t.pop()
    main_element.description = first_element["description"]
    main_element.words = [e["raw_keyword"] for e in first_element["seo"]]
    for element in t:
        elements.append(description_words(description=element["description"], words=[e["raw_keyword"] for e in element["seo"]]))

print(generate_new_text_with_seo_words_v2(main_element, elements))

