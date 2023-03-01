import os
import json
import keras
import numpy
import keras.preprocessing
import matplotlib.pyplot
import matplotlib.image

from keras.preprocessing.image import image_utils as keras_image_utils

class Prediction:
  def get_best_model(self, models_path):
    if not os.path.exists(models_path):
      print(f"[ERROR] Models folder {models_path} doesn't exist.")
      return None

    model_file = sorted(os.listdir(models_path))[-1]

    print(f"Best model: {model_file}")

    return f"{models_path}/{model_file}"
  
  def load_classes(self, classes_file):
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
  
  def predict(self, path, classes, model_file):
    results = []

    model = keras.models.load_model(model_file)

    if os.path.isdir(path):
      for image_path in os.scandir(path):
        results.append({image_path.path: classes[self._predict(model, image_path.path)]})
    else:
      results.append({path: classes[self._predict(model, path)]})
    return results
  
  def show_images(self, results):
    images_count = len(results)
    count = 1
    figure = matplotlib.pyplot.figure(figsize=(8, 8))
  
    for result in results:
      for image_path, image_class in result.items():
        axes = figure.add_subplot(round(images_count/3) + 1, 3, count)
        axes.axis('off')
        axes.imshow(matplotlib.image.imread(image_path))
        axes.set_title(image_class)
        count += 1 

    matplotlib.pyplot.show()
