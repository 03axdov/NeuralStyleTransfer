import tensorflow as tf
import tensorflow_hub as hub

def pretrained_model():
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    return hub_model

def vgg19(include_top=False):
    return tf.keras.applications.VGG19(include_top=include_top, weights="imagenet")

