import os
import json
import keras
import numpy
import cv2
import keras.preprocessing

from keras.preprocessing.image import image_utils as keras_image_utils


class Prediction:
    def __init__(self):
        self.__supported_formats = (".tiff", ".tif")
        self.__max_images = 4
        self.__taget_size = (150, 150)

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
    
    def remove_background(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = numpy.array([20, 30, 40])
        upper = numpy.array([100, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = numpy.ones((5,5), numpy.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        res = cv2.bitwise_and(image, image, mask=mask)
        bg = numpy.zeros_like(image)
        bg[mask != 0] = res[mask != 0]
        return bg

    def get_bonding_boxes(self, image):
        bonding_boxes = []

        image = self.remove_background(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        for contour in cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            (x, y, w, h) = cv2.boundingRect(contour)
            if (w > 30 and h > 30) and (w < 150 and h < 150):
                bonding_boxes.append({
                    "image": image[y:y+h, x:x+w],
                    "bbox": (x, y, w, h)
                })
        return bonding_boxes

    def _predict(self, model, image_path):
        print(f"Test image: {image_path}")
        predictions = []

        image = cv2.imread(image_path)

        for bbox in self.get_bonding_boxes(image):
            image = cv2.resize(
                bbox["image"], 
                self.__taget_size
            ).reshape((1,) + self.__taget_size + (3,))

            prediction = model.predict(image)[0]
    
            max_prob_index = numpy.argmax(prediction)
            probability = prediction[max_prob_index]*100

            if probability > 90:
                predictions.append({
                    "probability": probability,
                    "max_index": max_prob_index,
                    "bbox": bbox["bbox"]
                })

        return predictions

    def predict(self, path, model_file):
        results = []
        model = keras.models.load_model(model_file)

        if os.path.isdir(path):
            print(f"Runing prediction on folder {path}. Max images {self.__max_images}.")
            for image_path in os.scandir(path):
                if image_path.path.endswith(self.__supported_formats):
                    results.append([image_path.path, self._predict(model, image_path.path)])
                if len(results) >= self.__max_images:
                    break
        else:
            if path.endswith(self.__supported_formats):
                print(f"Runing prediction on file {path}")
                results.append([path, self._predict(model, path)])
            else:
                print(f"Not supported file format. Use one of {self.__supported_formats}")
        return results

    def save_results(self, path: str, results: list):
        with open(path, "w") as _file:
            _file.write(json.dumps(results))