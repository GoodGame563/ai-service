from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification

import os
import pika
import json
import requests
import grpc
import task_pb2
import task_pb2_grpc
import torch


class Element(BaseModel):
    url: str
    id: int



print(torch.cuda.is_available())
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_name = "miguelcarv/resnet-152-text-detector"
processor_name = "microsoft/resnet-50"

model_directory = f"{model_name.split('/')[1]}-ai"

try:
    model = AutoModelForImageClassification.from_pretrained(model_directory,).to(device)
    processor = AutoImageProcessor.from_pretrained(model_directory, do_resize=False)
except:
    model = AutoModelForImageClassification.from_pretrained(model_name,).to(device)
    model.save_pretrained(model_directory)
    processor = AutoImageProcessor.from_pretrained(processor_name, do_resize=False)
    processor.save_pretrained(model_directory)

def send_answer(success: bool, message:str, data: list = []):
    try:
        with grpc.insecure_channel(f"{os.getenv('GRPC_HOST')}:{os.getenv('GRPC_PORT')}") as channel:
            stub = task_pb2_grpc.TaskServiceStub(channel)
            request = task_pb2.CheckPhotoRequest(
                success=success,
                message=message,
                data=data
            )
            response = stub.CheckPhoto(request)
    except grpc.RpcError as e:
        print(f"Error connecting to gRPC server: {e}")
        return

def callback(ch, method, properties, body):
    message = Element(**json.loads(body))
    try:
        response = requests.get(message.url, timeout=3)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB').resize((300,300))
        inputs = processor(image, return_tensors="pt").to(device).pixel_values
        with torch.no_grad():
            outputs = model(inputs)
            
        logits_per_image = outputs.logits 
        logits_per_image = logits_per_image.cpu()
        probs = logits_per_image.softmax(dim=1) 
        send_answer(True, "All photos processed",[task_pb2.CheckPhotoData(id=message.id, value=bool(probs[0][0] < probs[0][1]))])
        return
    except UnidentifiedImageError:
        send_answer(False, f"Error: The URL did not return a valid image format. URL: {message.url}", [])
        return
    except requests.exceptions.HTTPError as http_err:
        send_answer(False, f"HTTP error occurred: {http_err} - URL: {message.url}", [])
        return
    except requests.exceptions.ConnectionError as conn_err:
        send_answer(False, f"Connection error occurred: {conn_err} - URL: {message.url}", [])
        return
    except requests.exceptions.Timeout as timeout_err:
        send_answer(False, f"Timeout error occurred: {timeout_err} - URL: {message.url}", [])
        return
    except Exception as ex:
        send_answer(False, f"Error processing message: {ex}", [])
        return   
    

def start_definition_text_consumer():
    credentials = pika.PlainCredentials(os.getenv('RABBITMQ_LOGIN'), os.getenv('RABBITMQ_PASSWORD'))
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.getenv('RABBITMQ_HOST'), os.getenv('RABBITMQ_PORT'), credentials=credentials))
    queue_name = 'ai_photo'
    channel = connection.channel()

    channel.exchange_declare(exchange=queue_name, exchange_type='direct', durable=False)
    channel.queue_declare(queue=queue_name, passive=True)

    channel.basic_consume(
        queue=queue_name,
        on_message_callback=callback,
        auto_ack=False 
    )
    print("Definition text consumer waiting for messages...")
    channel.start_consuming()

if __name__ == "__main__":
    start_definition_text_consumer()