import os
import json
import keras
import numpy
import cv2
import keras.preprocessing


class Prediction:
    """Prediction class."""

    def __init__(self):  # noqa
        self.__supported_formats = (".tiff", ".tif")
        self.__taget_size = (150, 150)

    def get_best_model(self, models_path: str) -> str:
        """Get best model from path, sorted by max accuracy.

        Args:
            models_path (str): path of models files.

        Returns:
            str: path of best model.
        """
        if not os.path.exists(models_path):
            print(f"[ERROR] Models folder {models_path} doesn't exist.")
            return None

        model_file = sorted(os.listdir(models_path))[-1]

        print(f"Best model: {model_file}")

        return f"{models_path}/{model_file}"

    def load_classes(self, classes_file: str) -> dict:
        """Load classes from json.

        Args:
            classes_file (str): json file with classes.

        Returns:
            dict: loaded classes.
        """
        if not os.path.exists(classes_file):
            print(f"[ERROR] File {classes_file} doesn't exist.")
            return None

        with open(classes_file) as _file:
            classes = json.loads(_file.read())

        return classes

    def remove_background(self, image: cv2.Mat) -> cv2.Mat:
        """Remove background from image.

        Args:
            image (cv2.Mat): image data array.

        Returns:
            cv.Mat: image data array.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = numpy.array([20, 30, 40])
        upper = numpy.array([100, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = numpy.ones((5, 5), numpy.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        res = cv2.bitwise_and(image, image, mask=mask)
        bg = numpy.zeros_like(image)
        bg[mask != 0] = res[mask != 0]
        return bg

    def get_bonding_boxes(self, image: cv2.Mat) -> list:
        """Get bonding boxes.

        Args:
            image (cv2.Mat): image data array.

        Returns:
            list: results.
        """
        bonding_boxes = []

        image = self.remove_background(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(
            gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        for contour in cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            (x, y, w, h) = cv2.boundingRect(contour)
            if (w > 15 and h > 15) and (w < 224 and h < 224):
                bonding_boxes.append({
                    "image": image[y:y+h, x:x+w],
                    "bbox": (x, y, w, h)
                })
        return bonding_boxes

    def _predict(self, model: any, image_path: str) -> list:
        """Prediction function.

        Args:
            model (cv2.Mat): model data.
            image_path (str): path image for prediction

        Returns:
            list: results.
        """
        print(f"Test image: {image_path}")
        predictions = []

        image = cv2.imread(image_path)
        bboxes = self.get_bonding_boxes(image)

        images = [
            cv2.resize(
                bbox["image"],
                self.__taget_size
            ).reshape((1,) + self.__taget_size + (3,))
            for bbox in bboxes
        ]

        if images:
            predictions = model.predict(numpy.vstack(images), use_multiprocessing=True)

            return [
                {
                    "probability": predictions[index][numpy.argmax(predictions[index])]*100,
                    "max_index": numpy.argmax(predictions[index]),
                    "bbox": bboxes[index]["bbox"]
                }
                for index in range(0, len(predictions))
                if predictions[index][numpy.argmax(predictions[index])]*100 > 90
            ]
        return []

    def predict(self, path: str, model_file: str) -> list:
        """Prediction function.

        Args:
            path (str): folder/image for prediction.
            model_file (str): path to model.

        Returns:
            list: results.
        """
        results = []
        model = keras.models.load_model(model_file, compile=False)

        if os.path.isdir(path):
            for image_path in os.scandir(path):
                if image_path.path.endswith(self.__supported_formats):
                    results.append(
                        [image_path.path, self._predict(model, image_path.path)])
        else:
            if path.endswith(self.__supported_formats):
                results.append([path, self._predict(model, path)])
            else:
                print(
                    f"Not supported file format. Use one of {self.__supported_formats}")
        return results
