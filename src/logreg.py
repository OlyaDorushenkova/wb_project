import json
from flask import Flask, request, jsonify
from PIL import Image
import io
from models import get_prediction
from preprocess import Clip_prep

def create_app(model_name, prep):
    app = Flask(__name__)
    @app.route('/', methods=['POST'])
    def predict():
        if request.method == 'POST':
            img_data = request.data
            img = Image.open(io.BytesIO(img_data))
            ans, proba = get_prediction(image = img,
                                        model_name = model_name,
                                        prep = prep)
            return jsonify({
                'class_id': ans.tolist(),
                'proba': proba
            })
    return app

app = create_app("./models/logreg", Clip_prep)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)