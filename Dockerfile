FROM python:3.9-slim-buster
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \ 
    unzip
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY src/ .
RUN wget -O ./models.zip "https://www.dropbox.com/scl/fi/c7zz1n5e9jjf7f1ymt3pz/models.zip?rlkey=2izpxyut0zka23naa7bik6kb0&st=z1p8p0cg&dl=1" && \
    unzip -x models.zip && \
    rm -r models.zip