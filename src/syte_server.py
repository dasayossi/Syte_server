import io
import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, jsonify, request
from tensorflow.keras.applications import ResNet50, imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array


def prepare_image(image, target=(224, 224)):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


model = ResNet50(weights="imagenet")

app = Flask('Syte_Server')

# http://localhost:5000/
@app.route("/predict",methods=["POST"])
def predict():
    app.logger.info(f'Processing prediction request - {datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")}')
    r = request
    try:
        image = Image.open(io.BytesIO(r.data))
        image = prepare_image(image)
        preds = model.predict(image)

        results = imagenet_utils.decode_predictions(preds)
        return jsonify(results.__str__())

    except Exception as e:
        app.logger.error(f'Failed processing request, type og error: {e.__str__}')
        return jsonify({'Failed': e.__str__})

@app.route('/health/')
def health():
    app.logger.info(f'Got health request - {datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")}')
    return {'message': 'Healthy'}  # This will return as JSON by default with a 200 status code


if __name__ == '__main__':
    app.run()
