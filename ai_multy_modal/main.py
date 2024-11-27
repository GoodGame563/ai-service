from PIL import Image
import os

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from threading import Thread
import requests
import asyncio
import aiohttp

import json
from pydantic import BaseModel

from tqdm import tqdm
import datetime
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Element(BaseModel):
    Images: str
    Objects_in_Image_by_Words: list
    Objects_in_Image_by_Words_EN: list
    Description_of_Image_in_Sentence: str
    Questions_for_Image: list
    Original_Description:str
    The_text_in_the_picture:list
    Words_in_the_picture:list
    Text_exsist:bool
    Human_exsist:bool


print(torch.cuda.is_available())
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

path_to_json = "../output.json"
model_name = "Qwen/Qwen2-VL-2B-Instruct"

model_directory = f"{model_name.split('/')[1]}"


try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_directory, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_directory)
except:
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(model_directory)
    model.save_pretrained(model_directory)


execution_times_sec = []
# model = model.to(device)
# model = CLIPModel.from_pretrained().to(device)
# processor = CLIPProcessor.from_pretrained()
   
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image 
# logits_per_image = logits_per_image.cpu()
    
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
print("end")

# with open(path_to_json, "r", encoding="utf-8") as f:
#     data = json.load(f)
#     for element in tqdm(data):
#         element = Element(**element)
#         start = datetime.datetime.now()



        
#         end = datetime.datetime.now()
#         execution_time = (end - start).total_seconds()
#         execution_times_sec.append(execution_time)



#     average_time_sec = sum(execution_times_sec) / len(data)
#     average_time_ms = average_time_sec * 1000
#     print(f"Среднее время выполнения datetime.now() в секундах: {average_time_sec:.10f} сек")
#     print(f"Среднее время выполнения datetime.now() в миллисекундах: {average_time_ms:.10f} мс")

