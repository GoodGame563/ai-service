FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu20.04
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем библиотеки Python, такие как numpy и tensorflow (или другие необходимые)
RUN pip3 install --upgrade pip

WORKDIR /app

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install grpcio==1.68.0, grpcio-tools==1.68.0

COPY . .

RUN python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. task.proto

RUN pip3 install -r requirements.txt


CMD ["python3", "main.py"] 