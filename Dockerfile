FROM python:3.9.17-bookworm

WORKDIR /app

RUN pip install pip-tools
COPY requirements.in requirements.in
RUN pip-compile -o requirements.txt requirements.in -v
RUN pip install --no-cache-dir -r requirements.txt
COPY . .