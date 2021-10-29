import time
import gzip, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageFilter


def save_run_history(history, save_path='./results/run_history.png'):
  """
    This helps to save the model training history results

    Parameter(s):
      history : history object
      save_path : where to save the reports
  """
  plt.figure(figsize=(14,8))
  plt.subplot(1, 2, 1)
  plt.suptitle('Training Results', fontsize=15)
  plt.ylabel('Loss', fontsize=12)
  plt.xlabel("#Epochs")
  plt.plot(history.history['loss'], color='blue', label='Training Loss')
  plt.plot(history.history['val_loss'], color='red', label='Validation Loss')
  plt.legend(loc='upper right')

  plt.subplot(1, 2, 2)
  plt.ylabel('Accuracy', fontsize=12)
  plt.xlabel("#Epochs")
  plt.plot(history.history['accuracy'], color='green', label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], color='orange', label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.savefig(save_path, facecolor='white')
  plt.close()
  print("Training history plots saved in results.")
  
  
def save_confusion_matrix(y_true, y_pred, class_labels, save_path='./results/confusion_matrix.png'):
  """
    This helps to save the confusion matrix for the trained model
  
    Parameter(s):
      y_true : true classes
      y_pred : predicted classes
      class_labels : class labels list, where index is the class
      save_path : where to save the reports
  """
  plt.figure(figsize=(14, 8))
  conf_matrix = np.round(confusion_matrix(y_true, y_pred, normalize='true'), 1)
  df_cm = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
  hMap = sns.heatmap(df_cm, annot=True)
  plt.xlabel('Predicted Label')
  plt.ylabel('Actual Label')
  plt.savefig(save_path, facecolor='white')
  plt.close()
  print("Confusion Matrix saved in results.")
  
  
def save_classification_report(y_true, y_pred, class_labels, save_path='./results/classification_report.txt'):
  """
    This helps to save the classification report for the trained model

    Parameter(s):
      y_true : true classes
      y_pred : predicted classes
      class_labels : class labels list, where index is the class
      save_path : where to save the reports
  """
  with open(save_path,'w') as cr:
    cr.write(classification_report(y_true, y_pred, target_names=class_labels))
  print("Classification report saved in results.")


def process_image(img_path, img_dims=(28,28)):
  """
    This function reads any image and processes it into a pixel values after converting 
    it into a grayscale version .

    Parameter(s):
      img_path : path of image file
      img_dims : dimensions pf image (width x height)
  """
  image = Image.open(img_path).convert('L')
  img_w, img_h = map(float, image.size)
  # creates white canvas of 28x28 pixels, or any other specified value
  new_image = Image.new('L', img_dims, (255))

  if img_w > img_h:
      # width is greater so it is fixed to 20 pixels and the height is adjusted acccordingly.
      new_w, new_h = 20, max(1, int(round((20.0 / img_w*img_h), 0)))
      # calculate horizontal and vertical positions
      wleft, wtop = 4, int(round(((img_dims[1] - new_h) / 2), 0))
  else:
      # height is greater/equal so it is fixed to 20 pixels and the width is adjusted acccordingly.
      new_w, new_h = max(1, int(round((20.0 / img_w*img_h), 0))), 20
      # calculate horizontal and vertical positions
      wleft, wtop = int(round(((img_dims[0] - new_w) / 2), 0)), 4

  img = image.resize((new_w, new_h), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
  # resized image is pasted after centering it on the newly created white canvas
  new_image.paste(img, (wleft, wtop))

  # get pixel values
  new_image_arr = list(new_image.getdata())
  return new_image_arr


def load_image_data(filepath):
  """
    Load the idx image data from given path

    Parameter(s):
      filepath : path to the image dataset file
  """
  with gzip.open(filepath, 'rb') as img_dataset:
    images = np.frombuffer(img_dataset.read(), dtype=np.uint8, offset=16)
  return images


def load_label_data(filepath):
  """
    Load the idx label data from given path

    Parameter(s):
      filepath : path to the label dataset file
  """
  with gzip.open(filepath, 'rb') as lb_dataset:
    labels = np.frombuffer(lb_dataset.read(), dtype=np.uint8, offset=8)
  return labels
