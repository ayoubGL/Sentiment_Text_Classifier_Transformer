FROM python:3.9-slim-bullseye

WORKDIR /app


# RUN apt-get update && \
#     # app-get install -y --no-install-recommends \
#     build-essential \
#     git \
#     libg11-mesa-glx \
#     libgsm6 \
#     libxext6 \
#     ffmpeg \
#     && rm -rf /var/lib.apt/lists*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV MLFLOW_TRAKING_URI=http://mlflow_server:5000


EXPOSE 8000


CMD ["python", "main.py"]