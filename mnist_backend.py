import keras
from flask import Flask
import numpy as np
import joblib
import cv2
import pickle
import tensorflow.compat.v2 as tf
from keras.models import model_from_json

app = Flask(__name__)

# with open('./mnist.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)
# model = joblib.load('mnist.joblib')
# model = load_model('mnist_h5')
json_file = open('myModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
new_model = model_from_json(loaded_model_json)


@app.route('/',methods = ['GET'])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
	return 'Hello World'


if __name__ == '__main__':
	app.run()
