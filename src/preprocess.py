import torch
import numpy as np
from clip import clip
from models import Basic_model
import easyocr

reader = easyocr.Reader(lang_list=['ru'], recognizer=True)

model_clip, preprocess_clip = clip.load("ViT-B/32")
model_clip.eval()

my_model_list = ["./models/logreg", "./models/SVC", "./models/NB"]

def Clip_prep(image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_input = preprocess_clip(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model_clip.encode_image(image_input)
    return np.array(image_features.cpu()).reshape(1, 512)

def OCR_prep(image):
    ans = np.array(reader.readtext(np.array(image), detail = 0, paragraph=True))
    text = ""
    if(len(ans) > 1):
        for i in range(len(ans)):
            text += ans[i]
            text += " "
    elif(len(ans) == 1):
        text = ans[0]
    return [text]

my_preprocess_list = [Clip_prep, Clip_prep, OCR_prep]

def stack_preprocess(image):
        ans = []
        for i in range(len(my_model_list)):
            helper = Basic_model(my_model_list[i], pretrained=True,
                                 preprocess = my_preprocess_list[i])
            ans.append(helper.predict(image)[0])
        return np.array(ans).reshape(1, -1)