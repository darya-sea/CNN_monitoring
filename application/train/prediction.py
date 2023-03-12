import os
import json
import keras
import numpy
import imutils
import cv2
import keras.preprocessing

from keras.preprocessing.image import image_utils as keras_image_utils
from keras.applications import imagenet_utils as kreas_imagenet_utils


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

    def crop_image(self, image, scale=1.5, minSize=(224, 224)):
        # yield the original image
        yield image
        # keep looping over the image pyramid
        while True:
            # compute the dimensions of the next image in the pyramid
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break
            # yield the next image in the pyramid
            yield image

    def sliding_window(self, image, step, ws):
        # slide a window across the image
        for y in range(0, image.shape[0] - ws[1], step):
            for x in range(0, image.shape[1] - ws[0], step):
                # yield the current window
                yield (x, y, image[y:y + ws[1], x:x + ws[0]])

    def _predict_v2(self, model, image_path):
        print(f"Test image: {image_path}")
        image = keras_image_utils.load_img(image_path, target_size=(224, 224))
        image = image.resize((600, image.size[1]))

        image = keras_image_utils.img_to_array(image)

        (H, W) = image.shape[:2]

        rois = []
        locs = []
        labels = []

        for image in self.crop_image(image):
            scale = W / float(image.shape[1])
            for (x, y, roiOrig) in self.sliding_window(image, 16, (300, 150)):
                # scale the (x, y)-coordinates of the ROI with respect to the
                # *original* image dimensions
                x = int(x * scale)
                y = int(y * scale)
                w = int((300, 150)[0] * scale)
                h = int((300, 150)[1] * scale)
                # take the ROI and preprocess it so we can later classify
                # the region using Keras/TensorFlow
                roi = cv2.resize(roiOrig, (224, 224))
                # roi = roiOrig.resize((224, 224))
                roi = keras_image_utils.img_to_array(roi)
                roi = kreas_imagenet_utils.preprocess_input(roi)
                # update our list of ROIs and associated coordinates
                rois.append(roi)
                locs.append((x, y, x + w, y + h))

        # image = numpy.expand_dims(image, axis=0)
        rois = numpy.array(rois, dtype="float32")

        preds = model.predict(rois)
        return preds
	
    def _predict(self, model, image_path):
        print(f"Test image: {image_path}")
        image = keras_image_utils.load_img(image_path, target_size=(224, 224))
        image = keras_image_utils.img_to_array(image)
        image = numpy.expand_dims(image, axis=0)
        return str(numpy.argmax(model.predict(image)))

    def predict(self, path, classes, model_file):
        results = []

        model = keras.models.load_model(model_file)

        if os.path.isdir(path):
            for image_path in os.scandir(path):
                if image_path.path.endswith((".tiff", ".tif", ".png", ".jpg", ".jpeg")):
                    results.append(
                        {image_path.path: classes[self._predict(model, image_path.path)]})
        else:
            if path.endswith((".tiff", ".tif", ".png", ".jpg", ".jpeg")):
                results.append({path: classes[self._predict(model, path)]})
        return results

    def save_results(self, path: str, results: list):
        with open(path, "w") as _file:
            _file.write(json.dumps(results))