import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = "COMPRESSED"

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.figsize"] = (6, 6)
mpl.rcParams["axes.grid"] = False

import numpy as np
import time
import functools
import PIL.Image

from utils import tensor_to_image, load_img, imshow
from models import *

def main():
    print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))}")

    content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # plt.subplot(1,2,1)
    # imshow(content_image, title="Content Image")
    #  
    # plt.subplot(1,2,2)
    # imshow(style_image, title="Style Image")
    # plt.show()

    # hub_model = pretrained_model()
    # stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    # img = tensor_to_image(stylized_image)
    # img.show()
    # img.save("Predictions/prediction1.jpg")

    # x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    # x = tf.image.resize(x, (224, 224))
    # vgg = vgg19(include_top=True)
    # prediction_probabilities = vgg(x)
    # print(f"Prediction probabilities: {prediction_probabilities.shape}")

    # predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    # print([(class_name, prob) for (number, class_name, prob) in predicted_top_5])

    vgg = vgg19()
    print()
    for layer in vgg.layers:
        print(layer.name)

if __name__ == "__main__":
    main()