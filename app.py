from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow and tf.keras

import tensorflow as tf
from tensorflow import keras

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
# sequence local
model = load_model('models/model_VGG16.h5')
CLASSES = ['cataract', 'glaucoma', 'diabetes', 'normal']
# sequence colab
CLASSES = ['normal', 'diabetes', 'glaucoma', 'cataract']

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/prediction', methods=['POST', 'GET'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "static/image.jpg" #+ imagefile.filename
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

    plt.bar(range(len(class_probabilities)), list(class_probabilities.values()), align='center')
    plt.xticks(range(len(class_probabilities)), list(class_probabilities.keys()))
    plt.plot()
    plt.savefig('/static/images/new_plot.png')

    
    return render_template('prediction.html', prediction =  class_probabilities, image=img, name = 'new_plot', url ='/static/images/new_plot.png')




if __name__ == '__main__':
    app.run(debug = True)