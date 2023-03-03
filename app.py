from flask import Flask, render_template, request
import numpy as np

# TensorFlow and tf.keras

import tensorflow as tf
from tensorflow import keras

""" from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions, ima_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image """


""" from keras.models import load_model
#from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import preprocess_input
from keras.preprocessing import image """

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions, ima_to_array



app = Flask(__name__)

model = load_model('models/model_VGG16_local.h5')
CLASSES = ['cataract', 'glaucoma', 'diabetes', 'normal']

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/"+ imagefile.filename
    imagefile.save(image_path)
    
    img = image.load_img(
        path=image_path,
        target_size=(224,224)
    )
    img_array = image.img_to_array(img)
    img_batch = np.array([img_array])
    preprocessed_img = preprocess_input(img_batch)
    probabilities = model.predict(
        preprocessed_img,
        verbose=0
    )
    probabilities = np.round(probabilities,3)[0]
    class_probabilities = dict(zip(CLASSES,probabilities))
    
    return render_template('index.html', prediction =  class_probabilities )




if __name__ == '__main__':
    app.run(debug = True)