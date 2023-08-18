import keras
from flask import Flask
import numpy as np
import joblib
import cv2


app = Flask(__name__)
model = joblib.load('mnist.joblib')
# model = load_model('mnist_h5')


@app.route('/',methods = ['GET'])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
	return 'Hello World'


if __name__ == '__main__':
	app.run()
