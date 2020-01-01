import numpy as np
from tensorflow.python.keras import layers, models
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# ensure consistency across runs
from numpy.random import seed

seed(1)


class AccuracyCallback(tf.keras.callbacks.Callback):

  def __init__(self, validation, target_accuracy=0.997):
    self.target_accuracy = target_accuracy
    self.validation = validation

  def on_epoch_end(self, epoch, logs):
    result = self.model.evaluate(self.validation)
    if result[-1] >= self.target_accuracy:
      print("val_loss: " + str(result[0]) + " val_acc: " + str(result[-1]))
      self.model.stop_training = True


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of the
        # dataset
        samplewise_std_normalization=False,  # divide each input by its std
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False #randomly flip images
)


def get_light_model(image_size):
  model = models.Sequential()
  model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
                          input_shape=image_size))
  model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(layers.Flatten())
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(512, activation="relu"))
  model.add(layers.Dropout(0.25))
  model.add(layers.Dense(10, activation="softmax"))
  return model


def train_generator(model,
        train_images, train_labels,
        validation_images, validation_labels,
        datagen=datagen,
        batch_size=128, epochs=10, verbose=1,
        optimizer="adam", loss_function="sparse_categorical_crossentropy",
        metrics=["accuracy"]):
  train_generator = datagen.flow(train_images, train_labels, batch_size,
                                 subset='training')

  validation_generator = datagen.flow(validation_images, validation_labels,
                                      batch_size=batch_size,
                                      subset='validation')

  model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

  history = model.fit_generator(
          generator=train_generator,
          validation_data=validation_generator,
          epochs=epochs,
          batch_size=batch_size,
          steps_per_epoch=train_images.shape[0] // batch_size,
          validation_steps=validation_images.shape[0] // batch_size,
          verbose=verbose)
  return model, history


def train_generator(model,
        train_images, train_labels,
        validation_images, validation_labels,
        datagen=datagen,
        batch_size=128, epochs=10,
        optimizer="adam", loss_function="sparse_categorical_crossentropy",
        metrics=['accuracy'], verbose=1):
  train_generator = datagen.flow(train_images.copy(), train_labels, batch_size,
                                 subset='training')

  validation_generator = datagen.flow(validation_images, validation_labels,
                                      batch_size=batch_size,
                                      subset='validation')
  model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

  history = model.fit_generator(generator=train_generator,
                                validation_data=validation_generator,
                                epochs=epochs,
                                steps_per_epoch=train_images.copy().shape[0] //
                                                batch_size,
                                validation_steps=validation_images.copy().shape[
                                                   0] //
                                                 batch_size,
                                verbose=verbose
                                ),
            
  return model, history

"""
train without datagerator                                                                                                                                                                                                                                                                                                             
"""
def train(
        model,
        train_images, train_labels,
        evaluate_images=None, evaluate_labels=None,
        batch_size=128,
        epochs=10,
        optimizer="adam",
        loss_function="sparse_categorical_crossentropy",
        metrics=["accuracy"]):
  model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
  if evaluate_images is None and evaluate_labels is None:
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
  else:
    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(evaluate_images, evaluate_labels),
                        batch_size=batch_size)
    return model, history
  return model


def test(model,
        test_images, test_labels):
  test_loss, test_acc = model.evaluate(test_images, test_labels,
                                       verbose=2)
  return test_loss, test_acc
