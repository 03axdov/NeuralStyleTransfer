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

from utils import *
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

    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    
    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255)

    for name, output in zip(style_layers, style_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()

    extractor = StyleContentModel(style_layers, content_layers)
    results = extractor(tf.constant(content_image))

    print("Styles:")
    for name, output in sorted(results['style'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results['content'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)

    opt = tf.keras.optimizers.Adam(learning_rate=0.02, epsilon=1e-1, beta_1=0.99)

    style_weight = 1e-2
    content_weight = 1e4

    def style_content_loss(outputs):
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                            for name in style_outputs.keys()])

        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                            for name in content_outputs.keys()])

        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    # for _ in range(5):
    #     train_step(image)

    # img = tensor_to_image(image)
    # img.show()


    start = time.time()

    epochs = 10
    steps_per_epoch = 100

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='', flush=True)

        print("Train step: {}".format(step))
    end = time.time()

    print("Total time: {:.1f}".format(end-start))
    image = tensor_to_image(image)
    image.show()


if __name__ == "__main__":
    main()