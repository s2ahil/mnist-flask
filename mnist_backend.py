import keras
from flask import Flask
import numpy as np
import joblib
import cv2
import pickle
import tensorflow.compat.v2 as tf
from keras.models import model_from_json

app = Flask(__name__)
model2=keras.models.load_model('model.h5')

@app.route('/', methods=['GET'])
def hello_world():
    json_file = open('myModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Read the uploaded image
    image_path = '8.png'  # Replace with the path to your image

    def preprocess():
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Resize the image to 28x28 pixels (MNIST size)
        resized_image = cv2.resize(gray_image, (28, 28))
        # Invert the colors (black to white, white to black)
        inverted_image = cv2.bitwise_not(resized_image)
        return inverted_image

    pre_img = preprocess()
    prediction = model.predict(np.array([pre_img]))
    predicted_digit = np.argmax(prediction)

    print(predicted_digit)
    return 'Hello World' + str(predicted_digit)

if __name__ == '__main__':
    app.run()
