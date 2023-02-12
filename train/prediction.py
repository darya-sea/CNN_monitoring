import os
import json
import keras
import numpy
import keras.preprocessing

from keras.preprocessing.image import image_utils as keras_image_utils

class Prediction:
  def __init__(self, data_folder):
    self.__data_folder = f"{data_folder}"
    self.__models_folder = f"{data_folder}/output/models"

  def load_model(self):
    if not os.path.exists(self.__models_folder):
      print(f"[ERROR] Models folder {self.__models_folder} doesn't exist.")
      return None

    model_file = sorted(os.listdir(self.__models_folder))[-1]

    print(f"Loaded best model: {model_file}")

    return keras.models.load_model(f"{self.__models_folder}/{model_file}")
  
  def load_classes(self):
    classes_file = f"{self.__data_folder}/validation_classes.json"

    if not os.path.exists(classes_file):
      print(f"[ERROR] File {classes_file} doesn't exist.")
      return None

    with open(classes_file) as _file:
      classes = json.loads(_file.read())

    return classes
  
  def _predict(self, model, image_path):

    print(f"Test image: {image_path}")
    image = keras_image_utils.load_img(image_path, target_size=(224, 224))
    input_image = keras_image_utils.img_to_array(image)
    input_image = numpy.expand_dims(input_image, axis=0)
    return str(numpy.argmax(model.predict(input_image)))
  
  def predict(self, path):
    results = []
    model =  self.load_model()
    classes = self.load_classes()

    if not model:
      print(f"[ERROR] Models not found.")
      return None

    if os.path.isdir(path):
      for image_path in os.scandir(path):
        results.append({image_path.name: classes[self._predict(model, image_path.path)]})
    else:
      results.append({os.path.basename(image_path): classes[self._predict(model, image_path)]})
    return results