FROM python:3.6.15-bullseye

COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install uvicorn[standard]
RUN pip install -r /requirements.txt

RUN pip install aiofiles

RUN pip install Jinja2

RUN pip install torchvision==0.9.1

RUN pip install vietocr==0.3.11

RUN pip install python-multipart

WORKDIR /app

COPY . /app/

EXPOSE 8000
