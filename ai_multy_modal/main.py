from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image, ImageFile
from io import BytesIO
from pydantic import BaseModel
from dotenv import load_dotenv

import sys
import os
import time

sys.path.append(os.path.join(os.getcwd(), '..'))
os.system('python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. task.proto')

load_dotenv()
import torch
import json
import requests
import grpc
import task_pb2
import task_pb2_grpc
import pika

class PhotoMessage(BaseModel):
    id: int
    url: str
    type: str

class PhotoMessageV2(BaseModel):
    id: int 
    product: list[str]
    competitors: list[list[str]]


print(torch.cuda.is_available())
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
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

model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
processor = LlavaNextProcessor.from_pretrained(model_name, do_resize=False)
model.config.pad_token_id = model.config.eos_token_id
model = torch.compile(model)



def request_to_multymodal(conversation: list, image:ImageFile):
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=800,  temperature=0.7, do_sample=True).cpu()

    return(str(processor.decode(output[0], skip_special_tokens=True)).split("assistant")[1])

def request_to_multymodal_V2(conversation: list, image:ImageFile, tokens: int):
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=tokens,  temperature=0.7, do_sample=True)

    return(str(processor.decode(output[0], skip_special_tokens=True)).split("assistant")[1])

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

def analyze_photo_by_fonts(message: PhotoMessage):
    result = []
    response = requests.get(message.url, timeout=3)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert('RGB')
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": "Оцени текст на изображении: насколько легко его прочитать? Укажи, виден ли текст четко, не сливается ли он с фоном."},
              {"type": "image"},
            ],
        },
    ]
    result.append(request_to_multymodal(conversation, image))
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": "Определи, подходит ли стиль шрифта данному изображению. Уместен ли он для карточки товара? Отнеси стиль шрифта к категории (например, строгий, декоративный, минималистичный)."},
              {"type": "image"},
            ],
        },
    ]
    result.append(request_to_multymodal(conversation, image))
    return result

def analyze_photo_by_quality(message: PhotoMessage):
    result = []
    response = requests.get(message.url, timeout=3)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert('RGB')
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": "Оцени, является ли фотография профессиональной (студийной): проверь качество света, ровность фона, резкость и симметрию"},
              {"type": "image"},
            ],
        },
    ]
    result.append(request_to_multymodal(conversation, image))
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": "Проанализируй фотографию на наличие теней, пересветов или размытости. Укажи, как можно улучшить эти параметры."},
              {"type": "image"},
            ],
        },
    ]
    result.append(request_to_multymodal(conversation, image))
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": "Оцени фон изображения. Он однотонный или загроможденный? Соответствует ли он студийному стандарту (например, белый или серый фон)?"},
              {"type": "image"},
            ],
        },
    ]
    result.append(request_to_multymodal(conversation, image))
    return result

def analyze_photo_to_text_optimization(message: PhotoMessage):
    result = []
    response = requests.get(message.url, timeout=3)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert('RGB')
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": "Рассмотри расположение текста на изображении. Укажи, гармонично ли он сочетается с визуальным фоном, не перекрывает ли ключевые элементы изображения."},
              {"type": "image"},
            ],
        },
    ]
    result.append(request_to_multymodal(conversation, image))
    return result

def analyze_all(message: PhotoMessageV2):
    from tqdm import tqdm
    promt = "Ты проффессиональный анализатор фотографий. Напиши следующие пункты которые ты видишь. 1. Есть ли текст на фотографии? 2. Если эта одежда если показ матерьяла? Если это техника если техническая информация? 3. Как виден человек в полный рост или нет? 4. Какие есть выделяющиеся части?"
    our_photos = []
    for i in tqdm(message.product):
        image = Image.open(requests.get(i, timeout=3, stream=True).raw)
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": promt},
                {"type": "image"},
                ],
            },
        ]
        text = request_to_multymodal_V2(conversation=conversation, image= image, tokens=400)
        our_photos.append(text)
        print(text)
    competitor_photos = []
    for m in tqdm(message.competitors):
        for i in tqdm(m):
            image = Image.open(requests.get(i, timeout=3, stream=True).raw)
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": promt},
                    {"type": "image"},
                    ],
                },
            ]
            text = request_to_multymodal_V2(conversation=conversation, image= image, tokens=400)
            print(text)
            competitor_photos.append(text)

    
    return (our_photos, competitor_photos)

def send_answer_to_fonts_analysis(success: bool, message:str, data: task_pb2.CheckFontsData):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.UpdateFontsAnalysisRequest(
                success=success,
                message=message,
                data=data
            )
            stub.UpdateFontsAnalysis(request)
    except Exception as e:
        print(f"Error: {e}")
        return
    
def send_answer_to_text_optimization(success: bool, message:str, data: task_pb2.CheckTextOptimizationData):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.UpdateTextOptimizationRequest(
                success=success,
                message=message,
                data=data
            )
            stub.UpdateTextOptimization(request)
    except Exception as e:
        print(f"Error: {e}")
        return
    
def send_answer_to_quality_analysis(success: bool, message:str, data: task_pb2.CheckQualityData):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.UpdateQualityAnalysisRequest(
                success=success,
                message=message,
                data=data
            )
            stub.UpdateQuanlityAnalysis(request)
    except Exception as e:
        print(f"Error: {e}")
        return

def send_answer_to_analyze_all(success: bool, message:str, data: task_pb2.PhotoAnalysisV3):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.UpdatePhotoAnalysisV3Request(
                success=success,
                message=message,
                data=data
            )
            stub.UpdatePhotoAnalysisV3(request)
    except Exception as e:
        print(f"Error: {e}")
        return

def callback(ch, method, properties, body):
    message = PhotoMessageV2(**json.loads(body))
    try:
        data = analyze_all(message)
        send_answer_to_analyze_all(True, "success generate",  task_pb2.PhotoAnalysisV3(
                id=message.id,
                ourPhotos=data[0],
                competitorPhotos=data[1],
                ))
    except Exception as e:
        send_answer_to_analyze_all(False, str(e), task_pb2.PhotoAnalysisV3(
                id=message.id,
                ourPhotos=[],
                competitorPhotos=[],
                ))
        print(f"Error analyze: {e}")
    finally:
        ch.basic_ack(delivery_tag=method.delivery_tag)


def start_seo_consumer():
    repit = 5
    # while True:
    #     try:
    credentials = pika.PlainCredentials(os.getenv('RABBITMQ_LOGIN'), os.getenv('RABBITMQ_PASSWORD'))
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.getenv('RABBITMQ_HOST'), os.getenv('RABBITMQ_PORT'), credentials=credentials,  heartbeat=60000))
    queue_name = 'photo_queue_v2'
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)

    channel.exchange_declare(exchange=queue_name, exchange_type='direct', durable=False)
    channel.queue_declare(queue=queue_name, passive=True)

    channel.basic_consume(
        queue=queue_name,
        on_message_callback=callback,
        auto_ack=False 
    )
    print("Definition photo consumer waiting for messages...")
    repit = 5
    channel.start_consuming()
            

        # except Exception as ex:
        #     print(f"Error starting consumer: {ex}")
        #     if repit == 0: 
        #         break
        #     print(f"Засыпаю на {1800/repit/60} минут")
        #     time.sleep(1800/repit)
        #     repit -= 1

if __name__ == "__main__":
    start_seo_consumer()
    pass