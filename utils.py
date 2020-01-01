import numpy as np
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

"""
summary data, get number samples and get size image
"""


def summary_data(data, label):
  y = np.bincount(label)
  ii = np.nonzero(y)[0]
  print("number samples and size each image: ", data.shape)
  print("counter label: \n", np.vstack((ii, y[ii])).T)


"""
to get sample datas by images and labels
"""


def get_sample_data(images, labels,
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


def show_history(history):
  plt.plot(history.history['acc'], label='acc')
  plt.plot(history.history['val_acc'], label='val_acc')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')


def show_sample_generator(generator):
  plt.gcf().set_size_inches(10, 10)
  for x_batch, y_batch in generator:
    for i in range(0, 25):
      plt.subplot(5, 5, i + 1)
      plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    break

# norm to [0 1]
def preprocessing_data(images):
  return (images / 255).reshape([-1, 28, 28, 1])

# expand dimension and norm [0 - 1]
def preprocessing_gan(images):
  X = np.expand_dims(images, axis=-1)
  X = X.astype('float32')
  X = X / 255.0
  return X


def confussion_matrix(y_test, y_prediction):
  fig = plt.figure(figsize=(10, 10))  # Set Figure
  Y_pred = np.argmax(y_prediction, 1)  # Decode Predicted labels
  Y_test = y_test
  mat = confusion_matrix(Y_test, Y_pred)  # Confusion matrix
  sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
  plt.xlabel('Predicted Values')
  plt.ylabel('True Values');
  plt.show()


def histogram_label(label):
  print("histogram - label -train")
  sns.countplot(label)
