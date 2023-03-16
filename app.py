from flask import Flask, render_template, request
import numpy as np

# deep learning stack
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


app = Flask(__name__)

model = load_model('models/model_VGG16_local_0503_cleaning.h5')
CLASSES = ['cataract', 'glaucoma', 'diabetes', 'normal']


@app.route('/', methods=['GET'])
def start_page():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():

    if request.form['selected_image']: # image via gallery selection
        image_path = request.form['selected_image'].replace('/', '', 1)
    # if request.files['imagefile']: # image via file upload
        # imagefile = request.files['imagefile']
        # image_path = "static/assets/image.jpg"
        # imagefile.save(image_path)
    else:
        return render_template('index.html')
    
    # get age and sex values
    ageValue = request.form['age']
    sexValue = request.form['sex']

    
    # image loading and preprocessing 
    img = image.load_img(
        path= image_path,
        target_size=(224,224)
    )
    img_array = image.img_to_array(img)
    img_batch = np.array([img_array])
    preprocessed_img = preprocess_input(img_batch)

    # prediction
    probabilities = model.predict(
        preprocessed_img,
        verbose=0
    )
    probabilities = np.round(probabilities,3)[0]

    # save CLASSES and probabilities in dictionary 
    class_probabilities = dict(zip(CLASSES,probabilities))

    # get specific values for each class 
    probability_cataract = probabilities[0]
    probability_glaucoma = probabilities[1]
    probability_diabetes = probabilities[2]
    probability_normal = probabilities[3]


    return render_template('index.html', 
                           age = ageValue,
                           sex = sexValue,
                           prediction = class_probabilities, 
                           probability_normal = probability_normal,
                           probability_diabetes = probability_diabetes,
                           probability_glaucoma = probability_glaucoma,
                           probability_cataract = probability_cataract,
                           image=image_path
                           )



if __name__ == '__main__':
    app.run()