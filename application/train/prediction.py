import os
import json
import keras
import numpy
import keras.preprocessing

from keras.preprocessing.image import image_utils as keras_image_utils
from keras.applications import imagenet_utils


class Prediction:
    def __init__(self):
        self.__supported_formats = (".tiff", ".tif", ".png", ".jpg", ".jpeg")

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
        image = keras_image_utils.img_to_array(image)
        image = numpy.expand_dims(image, axis=0)

        (box_preds, label_preds) = model.predict(image)
        (x, y, w, h) = box_preds[0]

        return int(x), int(y), int(w), int(h), str(numpy.argmax(label_preds))

    def predict(self, path, model_file):
        results = []

        model = keras.models.load_model(model_file)

        if os.path.isdir(path):
            print(f"Runing prediction on folder {path}")
            for image_path in os.scandir(path):
                if image_path.path.endswith(self.__supported_formats):
                    results.append([image_path.path, *self._predict(model, image_path.path)])
        else:
            if path.endswith(self.__supported_formats):
                print(f"Runing prediction on file {path}")
                results.append([path, *self._predict(model, path)])
            else:
                print(f"Not supported file format. Use one of {self.__supported_formats}")
        return results

    def save_results(self, path: str, results: list):
        with open(path, "w") as _file:
            _file.write(json.dumps(results))