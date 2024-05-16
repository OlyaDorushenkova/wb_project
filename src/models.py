from joblib import load
import numpy as np

class Basic_model:
    def __init__(self, 
                 model_name, 
                 pretrained = True, 
                 model = None,
                 preprocess = None):
        if pretrained:
            self.model = load(model_name + '.joblib')
        else:
            self.model = model
        self.preprocess = preprocess 
        
    def predict(self, image):
        return self.model.predict(self.preprocess(image))
        
    def predict_proba(self, image):
        pred = self.model.predict_proba(self.preprocess(image))
        return max(pred[0])
    
def get_prediction(image, model_name, prep):
    new_model = Basic_model(model_name = model_name, preprocess = prep)
    return new_model.predict(image), new_model.predict_proba(image)

    