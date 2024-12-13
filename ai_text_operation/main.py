from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.models as models
from pydantic import BaseModel
from dotenv import load_dotenv

import sys
import os

sys.path.append(os.path.join(os.getcwd(), '..'))
os.system('python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. task.proto')


import grpc
import task_pb2
import task_pb2_grpc
import pika
import torch
import json


load_dotenv()

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

# text = "Профессиональный спрей кондиционер для волос ULAB станет незаменимым помощником в ежедневном уходе за волосами. Этот продукт обеспечивает надежную термозащиту, разглаживание, увлажнение и восстановление. \n\nСпрей для волос обеспечивает защиту при укладке, благодаря чему волосы остаются гладкими и мягкими. Термозащитный эффект спрея предотвращает повреждение волос, сохраняя их силу и блеск. Этот спрей также защищает волосы от УФ-лучей.\n\nСредство солнцезащитное способствует легкому расчесыванию и укладке, благодаря чему волосы становятся более послушными. Спрей придает объем у корней для прикорневого объема, делая волосы густыми. Если ваши волосы склонны к пушистости или секущимся кончикам, этот спрей поможет разгладить их и придать волосам ухоженный вид. Эффект ламинирования, который дает спрей, делает волосы шелковистыми и блестящими.\n\nПарфюмированный спрей идеально подходит для использования на поврежденных, тонких и вьющихся волосах, обеспечивая им дополнительное питание и увлажнение, и защиту от солнца. Кератин в составе спрея способствует восстановлению структуры волос.\n\nЭтот кератиновый разглаживающий спрей для волос подходит для ухода за кожей головы, предотвращая сухость и шелушение. Он обеспечивает комплексный уход, питая волосы от корней до кончиков. \n\nЖенский спрей станет отличным выбором для тех, кто стремится к салонному уходу, к гладкости и блеску волос. Станет прекрасным подарком для женщин и девушек.\n\nСухие, поврежденные, пушистые или тонкие - многофункциональный спрей юлаб 10 в 1 обеспечит уход и восстановление. \n\nПрофессиональный несмываемый термоспрей для волос, особенно для окрашенных, обеспечивает защиту и эластичность, предотвращая электризацию и антистатик. Он идеален после утюжка и фена, сохраняя эффект кератинового выпрямления. Этот бьюти продукт незаменим для здорового и красивого вида волос."
# generate_new_text_without_seo_words(text)
# print(len(text.split()))
# t, f = (0, 0)
# for _ in range(50):
#     if text_write_rigth(text): t += 1
#     else: f += 1

# print(f"True: {t}, False: {f}")

# main_element = description_words(description="",words=[])
# elements = []
# with open("response.json", "r", encoding='utf-8') as write_file:
#     t = list(json.load(write_file))
#     words = []
    
#     first_element = t.pop()
#     main_element.description = first_element["description"]
#     main_element.words = [e["raw_keyword"] for e in first_element["seo"]]
#     for element in t:
#         elements.append(description_words(description=element["description"], words=[e["raw_keyword"] for e in element["seo"]]))
# print(generate_new_text_with_seo_words_v2(main_element, elements))

# with open("response.json", "r", encoding='utf-8') as write_file:
#     t = list(json.load(write_file))
#     first_element = t.pop()
#     reviews = first_element['reviews']
#     print(generate_by_reviews(reviews))

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



def callback(ch, method, properties, body):
    raw_type_message = json.loads(body)
    print("callback")

    if str(raw_type_message['type']) == 'reviews':
        try:
            message = ReviewsMessage(**json.loads(body))
            result = generate_by_reviews(message.reviews)
            send_answer_to_reviews(True, "All reviews processed", task_pb2.CheckReviewsData(
                id=message.id,
                value=result
                ))
        except Exception as ex:
            send_answer_to_reviews(False, f"Error generating reviews message: {ex}", task_pb2.CheckReviewsData(
                id=message.id,
                value=""
                ))
            print("Error")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        
    elif str(raw_type_message['type']) == 'seo_v1' and str(raw_type_message['type']) == 'seo_v2':
        try:
            message = SeoMessage(**json.loads(body))
            result = ""
            if message.type == 'seo_v1':
                result = generate_new_text_with_seo_words_v1(message.product, message.competitors)
            else:
                result = generate_new_text_with_seo_words_v2(message.product, message.competitors)

            send_answer_to_description(True, "Description processed",task_pb2.CheckDescriptionData(
                id=message.id,
                value=result
                ))
        except Exception as ex:
            send_answer_to_description(False, f"Error generating description message: {ex}", task_pb2.CheckDescriptionData(
                id=message.id,
                value=""
                ))
            print("Error")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return   
    print("Done")


def start_seo_consumer():
    credentials = pika.PlainCredentials(os.getenv('RABBITMQ_LOGIN'), os.getenv('RABBITMQ_PASSWORD'))
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.getenv('RABBITMQ_HOST'), os.getenv('RABBITMQ_PORT'), credentials=credentials))
    queue_name = 'seo_queue'
    channel = connection.channel()

    channel.exchange_declare(exchange=queue_name, exchange_type='direct', durable=False)
    channel.queue_declare(queue=queue_name, passive=True)

    channel.basic_consume(
        queue=queue_name,
        on_message_callback=callback,
        auto_ack=False 
    )
    print("Definition seo consumer waiting for messages...")
    channel.start_consuming()

if __name__ == "__main__":
    start_seo_consumer()