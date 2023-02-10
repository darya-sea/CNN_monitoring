import os
import keras
import numpy
import keras.preprocessing
import json

from keras.preprocessing.image import image_utils as keras_image_utils

class Prediction:
  def __init__(self, data_folder):
    self.__data_folder = f"{data_folder}"
    self.__output_folder = f"{data_folder}/output"

  def load_model(self):
    if not os.path.exists(self.__output_folder):
      print(f"[ERROR] Models folder {self.__output_folder} doesn't exist.")
      return None

    model_file = sorted(os.listdir(self.__output_folder))[-1]

    print(f"Loaded best model: {model_file}")

    return keras.models.load_model(f"{self.__output_folder}/{model_file}")
  
  def load_classes(self):
    if not os.path.exists(self.__output_folder):
      print(f"[ERROR] Models folder {self.__output_folder} doesn't exist.")
      return None

    with open(f"{self.__data_folder}/validation_classes.json") as _file:
      classes = json.loads(_file.read())

    return classes
  
  def predict(self, image_path):
    model =  self.load_model()

    if not model:
      print(f"[ERROR] Couldn't load model from {image_path}")
      return None

    print(f"Test image: {image_path}")
    image = keras_image_utils.load_img(image_path, target_size=(224,224))
    input_image = keras_image_utils.img_to_array(image)
    input_image = numpy.expand_dims(input_image, axis=0)

    classname = self.load_classes()[str(numpy.argmax(model.predict(input_image)))]

    return classname