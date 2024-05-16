# Модель классификации изображений в отзывах WB с помощью алгоритмов машинного обучения
### 1. Описание модели
Модель состоит из трех простых алгоритмов машинного обучения, соединенных с помощью стэккинга с случайным лесом, как итоговым классификатором.

Для предобработки изображений использовались предобученные модели: CLIP (для SVC, Logreg был использован предобученный эмбеддинг) и EasyOCR (для распознования текста на изображениях и классификации данных с помощью NB). 

Алгоритмы в стеккинге: логистическая регрессия, SVC и Naive Byess

### 2. Инструкция для запуска:
Скопировать репозиторий и собрать контейнеры:
```
git clone repo_name
docker compose up -d --build
```

Пример для запроса:

```
from PIL import Image
import json
import requests
import io


def image_to_byte_array(image: Image) -> bytes:
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format=image.format)
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


urls = {
    «clip_svm»: "http://172.16.238.11:5001",
    «clip_log_reg»: "http://172.16.238.12:5002",
    «ocr_NB»: "http://172.16.238.10:5000",
    «stacking»: "http://172.16.238.13:5003",
}

img = Image.open("test.jpeg")
img_bytes = image_to_byte_array(img)

response = requests.post(urls["whole_pipeline"], data = img_bytes)

print(json.loads(response.content.decode()))
```
Адресса моделей:

1) SVC - http://172.16.238.11:5001
2) LogReg - http://172.16.238.12:5002
3) NB - http://172.16.238.10:5000
4) Stacking: [NB + LogReg + NB] + RandomForest - http://172.16.238.13:5003

Ссылка на архив с предобученными моделями: https://www.dropbox.com/scl/fi/c7zz1n5e9jjf7f1ymt3pz/models.zip?rlkey=2izpxyut0zka23naa7bik6kb0&e=2&st=uhxob2qo&dl=0

Ответ представляется в виде класса и вероятность принадлежности к нему 

{'class_id': [0], 'proba': 0.6415903472216106} 
 
 Запрос обязан содержать 1 картинку.





