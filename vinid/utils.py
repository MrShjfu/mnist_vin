import numpy as np
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from numpy.random import randint


def summary_data(data, label):
  y = np.bincount(label)
  ii = np.nonzero(y)[0]
  print("number samples and size each image: ", data.shape)
  print("counter label: \n", np.vstack((ii, y[ii])).T)


def get_sample_data(
        images,
        labels,
        class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
  plt.figure(figsize=(10, 10))
  for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels[i]])
  plt.show()


def get_light_model(image_size):
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation=
  "relu", input_shape=image_size))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation="relu"))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation="relu"))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation="relu"))
  model.add(layers.Dense(10, activation="softmax"))
  # model.summary()
  return model


def train(
        model,
        train_images,
        train_labels,
        evaluate_images=None,
        evaluate_labels=None,
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


def test(
        model,
        test_images,
        test_labels):
  test_loss, test_acc = model.evaluate(test_images, test_labels,
                                       verbose=2)
  return test_loss, test_acc


def show_history(history):
  plt.plot(history.history['acc'], label='acc')
  plt.plot(history.history['val_acc'], label='val_acc')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')


def preprocessing_data(images):
  return (images / 255).reshape([-1, 28, 28, 1])


def preprocessing_gan(images):
  X = np.expand_dims(images, axis=-1)
  X = X.astype('float32')
  X = X / 255.0
  return X
