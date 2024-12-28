from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from dotenv import load_dotenv

import sys
import os
import time
import requests

sys.path.append(os.path.join(os.getcwd(), '..'))
os.system('python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. task.proto')


import grpc
import task_pb2
import task_pb2_grpc
import pika
import torch
import json


load_dotenv()
class PhotoReport(BaseModel):
    id: int 
    type: str
    our_photos: list[str]
    competitor_photos: list[str]

class ReviewsProductItemV2(BaseModel):
    name: str
    reviews_url: str

class SeoProductItem(BaseModel):
    name: str
    description: str

class SeoWord(BaseModel):
    keyword_id: str
    raw_keyword: str
    freq: int
    position: int

class Product(BaseModel):
    id: int 
    name: str
    description: str
    seo: list[SeoWord] 
    
class SeoMessage(BaseModel):
    id: int
    type: str
    product: Product
    competitors: list[Product]

class ReviewsMessage(BaseModel):
    id: int
    type: str
    reviews: list[str]

class ReviewsMessageV2(BaseModel):
    id: int
    type: str
    product: ReviewsProductItemV2
    competitors: list[ReviewsProductItemV2]

class SEOMessageV2(BaseModel):
    id:int 
    type: str
    product: SeoProductItem
    competitors: list[SeoProductItem]




model_directory = "./qwen_model-ai"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory)
except:
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    local_directory = "./qwen_model"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.save_pretrained(local_directory)
    model.save_pretrained(local_directory)

print(torch.cuda.is_available())
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

device = torch.device("cuda:0")
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

def generate_new_text_with_seo_words_v1(main_element:Product, elements:list[Product]) -> str:
    prompt = f"Прочитай описание нашего товара и список ключевых слов:\nНаше описание: {main_element.description}\nНаши ключевые слова: {[word.raw_keyword for word in main_element.seo]}\nПрочитай описание и ключевые слова конкурентов (без упоминания их брендов):"
    for element in  elements:
        prompt += f"\nОписание конкурента: {element.description}\nКлючевые слова конкурента: {[word.raw_keyword for word in element.seo]}"
    
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

def generate_photo_analysis(products: PhotoReport) -> str:
    print (products.our_photos)
    print (products.competitor_photos)
    prompt = f"Наши ответы на вопросы:\n\n"
    for product in products.our_photos:
        prompt += product + "\n"
    prompt += "Ответы на вопросы конкурентов: "
    for element in  products.competitor_photos:
        prompt += element + "\n"
    messages = [
        { "content": "Ты аналитик, тебе будут присланы ответы на данные вопросы 1. Есть ли текст на фотографии? 2. Если эта одежда если показ матерьяла? Если это техника если техническая информация? 3. Как виден человек в полный рост или нет? 4. Какие есть выделяющиеся части?. Это ответы на вопросы от другой нейросети тебе нужно их проанализировать и написать что необходимо добавить нашему товару посравнению с конкурентами. Пиши все чего не хватает, а если такого нет выдай свои рекомендации на основе конкурентов."},
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
        temperature=0.7,
        max_new_tokens=6096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print("анализ описания")
    print(final)
    return final

def generate_new_text_with_seo_words_v2(main_element:Product, elements:list[Product]) -> str:
    prompt = f"Прочитай описание нашего товара и список ключевых слов:\nНаше описание: {main_element.description}\nНаши ключевые слова: {[word.raw_keyword for word in main_element.seo]}\nПрочитай описание и ключевые слова конкурентов (без упоминания их брендов):"
    for element in  elements:
        prompt += f"\nОписание конкурента: {element.description}\nКлючевые слова конкурента: {[word.raw_keyword for word in element.seo]}"
    
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

def generate_new_text_with_concurent_text(main_element:SeoProductItem, elements:list[SeoProductItem]) ->str:
    prompt = f"Данные для анализа:\n\n1. Наш товар:\n- Название: {main_element.name}\n- Описание: {main_element.description}\n\n2. Товары конкурентов:"
    for element in  elements:
        prompt += f"\nНазвание: {element.name}\nОписание: {element.description}\n"
    messages = [
        {"role": "system", "content": "Ты — аналитик по улучшению описаний товаров для маркетплейсов. Твоя задача — анализировать тексты, находить тенденции в описаниях конкурентов и давать рекомендации для улучшения описания нашего товара. Все выводы должны быть четкими, профессиональными и следовать единому стилю. Выводы структурируй в формате: Выявленные тенденции, Рекомендации по улучшению описания, Конкретные изменения в нашем описании чтоб оно стало больше похоже на описание конкурентов. Все твои рекомендации должны быть направленны на улучшение только текстовой составлящей нашего товара."},
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
        temperature=1.0,
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print("анализ описания")
    # print(final)
    return final

def generate_by_reviews(reviews:list[str]) -> str:
    reviews_str = ""
    for review in reviews:
        reviews_str += f"{review}, "
    prompt = f"Отзывы {reviews_str}"
    messages = [
        {"role": "system", "content": """Ты проффесиональный анализатор отзывов о товаре на маркетплейсе
         Проанализируй отзывы покупателей на товары конкурентов. Определи 10 наиболее часто упоминаемых преимуществ и основываясь на отзывах товара. Опиши 10 ключевых характеристик и 10 преимуществ, которые делают их популярными среди пользователей. 
         Выпиши, что больше всего не нравится клиентам в использовании товара. 
         Напиши список того, что больше всего не нравится теми словами, как пишут в отзывах, расположив популярность от популярных к менее популярным. 
         После завершения анализа, выдели сегменты целевой аудитории на основе ответов из отзывов"""},
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
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return final

def generate_by_reviews_v2(reviews:ReviewsMessageV2) -> str:
    final = ""
    request = requests.get(reviews.product.reviews_url)
    r_reviews = request.text
    for r in reviews.competitors:
        prompt = f"Данные для анализа:\n\n1. Наше название :\n {reviews.product.name} \n Наши отзывы{r_reviews}"
        request = requests.get(r.reviews_url)
        r_r = request.text
        prompt += f"Название конкурентов {r.name}\nОтзывы конкурентов {r_r}"


        messages = [
            {"role": "system", "content": """Вот список отзывов о нашем продукте и продуктах конкурентов на маркетплейсе. Проанализируй их и предоставь следующие данные:

                Что хвалят в продуктах конкурентов? Перечисли основные достоинства, которые отмечают пользователи.
                Какие недостатки конкурентов упоминаются наиболее часто?
                Что пользователи хвалят в нашем продукте?
                Какие недостатки упоминаются в отзывах о нашем продукте?
                На основании анализа, предложи идеи для улучшения нашего товара, чтобы он стал привлекательнее для покупателей, учитывая сильные стороны конкурентов.
                Формат ответа:
                1. Достоинства конкурентов:

                [Описание достоинства 1]
                [Описание достоинства 2]
                2. Недостатки конкурентов:

                [Описание недостатка 1]
                [Описание недостатка 2]
                3. Достоинства нашего товара:

                [Описание достоинства 1]
                [Описание достоинства 2]
                4. Недостатки нашего товара:

                [Описание недостатка 1]
                [Описание недостатка 2]
                5. Рекомендации по улучшению:

                [Описание рекомендации 1]
                [Описание рекомендации 2]"""},
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
        final += "Анализ"
        final += tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print("анализ отзывов")
    # print(final)
    return final

def send_answer_to_description(success: bool, message:str, data: task_pb2.CheckDescriptionData):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.UpdateDescriptionRequest(
                success=success,
                message=message,
                data=data
            )
            stub.UpdateDescription(request)
    except grpc.RpcError as e:
        print(f"Error connecting to gRPC server: {e}")
        return
    
def send_answer_to_reviews(success: bool, message:str, data: task_pb2.CheckReviewsData):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.UpdateReviewsAnalyzeRequest(
                success=success,
                message=message,
                data=data
            )
            stub.UpdateReviewsAnalyze(request)
    except grpc.RpcError as e:
        print(f"Error connecting to gRPC server: {e}")
        return

def send_answer_to_description_v2(success: bool, message:str, data: task_pb2.SEOAnalysisV2):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.UpdateSEOAnalysisV2Request(
                success=success,
                message=message,
                data=data
            )
            stub.UpdateSEOAnalysisV2(request)
    except grpc.RpcError as e:
        print(f"Error connecting to gRPC server: {e}")
        return

def send_answer_to_reviews_v2(success: bool, message:str, data: task_pb2.ReviewsAnalysisV2):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.UpdateReviewsAnalysisV2Request(
                success=success,
                message=message,
                data=data
            )
            stub.UpdateReviewsAnalysisV2(request)
    except grpc.RpcError as e:
        print(f"Error connecting to gRPC server: {e}")
        return

def send_answer_to_analys_all(success: bool, message:str, data: task_pb2.PhotoAnalysisV2):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.UpdatePhotoAnalysisV2Request(
                success=success,
                message=message,
                data=data
            )
            stub.UpdatePhotoAnalysisV2(request)
    except grpc.RpcError as e:
        print(f"Error connecting to gRPC server: {e}")
        return
def callback(ch, method, properties, body):
    raw_type_message = json.loads(body)
    if str(raw_type_message['type']) == 'reviews':
        try:
            print(json.loads(body))
            message = ReviewsMessageV2(**json.loads(body))
            
            result = generate_by_reviews_v2(message)
            send_answer_to_reviews_v2(True, f"Success", task_pb2.ReviewsAnalysisV2(
                id=message.id,
                value=result
                ))
        except Exception as ex:
            print(ex)
            send_answer_to_reviews_v2(False, f"Error generating reviews message: {ex}", task_pb2.ReviewsAnalysisV2(
                id=message.id,
                value=""
                ))
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return 
    elif str(raw_type_message['type']) == 'seo':
        try:
            message = SEOMessageV2(**json.loads(body))
            result = generate_new_text_with_concurent_text(message.product, message.competitors)
            send_answer_to_description_v2(True,  f"Success", task_pb2.SEOAnalysisV2(
                id=message.id,
                value=result
                ))
        except Exception as ex:
            print(ex)
            send_answer_to_description_v2(False, f"Error generating reviews message: {ex}", task_pb2.SEOAnalysisV2(
                id=message.id,
                value=""
                ))
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return 
    elif str(raw_type_message['type']) == 'photo_report':
        try:
            message = PhotoReport(**json.loads(body))
            result = generate_photo_analysis(message)
            send_answer_to_analys_all(True,  f"Success", task_pb2.PhotoAnalysisV2(
                id=message.id,
                value=result
                ))
            print (result)
        except Exception as ex:
            print(ex)
            send_answer_to_description_v2(False, f"Error generating reviews message: {ex}", task_pb2.SEOAnalysisV2(
                id=message.id,
                value=""
                ))
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return 

def start_seo_consumer():
    repit = 5
    while True:
        try:
            credentials = pika.PlainCredentials(os.getenv('RABBITMQ_LOGIN'), os.getenv('RABBITMQ_PASSWORD'))
            connection = pika.BlockingConnection(pika.ConnectionParameters(os.getenv('RABBITMQ_HOST'), os.getenv('RABBITMQ_PORT'), credentials=credentials,  heartbeat=60000))
            queue_name = 'seo_queue_v2'
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)

            channel.exchange_declare(exchange=queue_name, exchange_type='direct', durable=False)
            channel.queue_declare(queue=queue_name, passive=True)

            channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback,
                auto_ack=False 
            )
            print("Definition seo consumer waiting for messages...")
            repit = 5
            channel.start_consuming()            
        except Exception as ex:
            print(f"Error starting consumer: {ex}")
            if repit == 0: 
                break
            print(f"Засыпаю на {1800/repit/60} минут")
            time.sleep(1800/repit)
            repit -= 1

if __name__ == "__main__":
    start_seo_consumer()
    pass