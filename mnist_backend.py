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
# model = joblib.load('mnist.pkl')
# model = load_model('mnist_h5')
# json_file = open('myModel.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# new_model = model_from_json(loaded_model_json)


@app.route('/',methods = ['GET'])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
	json_file = open('myModel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        new_model = model_from_json(loaded_model_json)
	# Read the uploaded image
        image_path = '8.png'  # Replace with the path to your image
        def preprocess():
         gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
         # Resize the image to 28x28 pixels (MNIST size)
         resized_image = cv2.resize(gray_image, (28, 28))
         # Invert the colors (black to white, white to black)
         inverted_image = cv2.bitwise_not(resized_image)

         # Display the inverted grayscale image with 'gray' colormap
 # plt.figure(figsize=(5, 5))
 # plt.title('Inverted Grayscale Image')
 # plt.imshow(inverted_image, cmap='gray')  # Use 'gray' colormap
 # plt.axis('off')

 # plt.tight_layout()
 # plt.show()
         return inverted_image 
        pre_img=preprocess()
	prediction = model1.predict(np.array([pre_img]))
        predicted_digit = np.argmax(prediction)

        print(predicted_digit)
	return 'Hello World'+str(predicted_digit)


if __name__ == '__main__':
	app.run()
