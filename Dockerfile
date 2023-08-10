# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM python:3.9-bookworm

WORKDIR /app

COPY . .
# Install dependencies
RUN pip install pip-tools
# Uncomment this, just takes too long to compile
# RUN pip-compile -o requirements.txt requirements.in -v
RUN pip install -r requirements.txt
# Install tsboi as package
RUN python3 setup.py install