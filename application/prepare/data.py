import shutil
import os
import numpy
import random
import cv2
import tempfile


class PrepareData:
    def __init__(self, input_folder, output_folder):
        self.__input_folder = input_folder
        self.__output_folder = output_folder

    def remove_background(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = numpy.array([20, 50, 50])
        upper = numpy.array([100, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = numpy.ones((5,5), numpy.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        res = cv2.bitwise_and(image, image, mask=mask)
        bg = numpy.zeros_like(image)
        bg[mask != 0] = res[mask != 0]
        return bg

    def get_all_images(self, folder):
        images = []

        for entry in os.scandir(folder):
            if entry.is_dir():
                images.extend(self.get_all_images(entry.path))
            else:
                if entry.path.endswith(("C.tiff", "C.tif")):
                    images.append(entry.path)

        return images

    def make_annotations(self, csv_file, output_image):
        plant_name = os.path.basename(os.path.dirname(output_image))

        if os.path.exists(output_image):
            image = cv2.imread(output_image)
            orignal_image = image.copy()

            image = self.remove_background(image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            with open(csv_file, "a") as _file:
                for contour in cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if w > 10 or h > 10:
                        cv2.rectangle(orignal_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        _file.write(f"{output_image},{x},{y},{w},{h},{plant_name}\n")
            
            cv2.imwrite(output_image, orignal_image)

    def prepare_images(self):
        if self.__input_folder == self.__output_folder:
            print(
                f"[ERROR] Input and output folders are same: {self.__input_folder}. Stopped.")
            return False

        shutil.rmtree(self.__output_folder, ignore_errors=True)

        train_annotations = os.path.join(self.__output_folder, "train_annotations.csv")
        validation_annotations = os.path.join(self.__output_folder, "validation_annotations.csv")

        for plant_name in os.listdir(self.__input_folder):
            if not os.path.isdir(os.path.join(self.__input_folder, plant_name)):
                continue

            train_folder = os.path.join(self.__output_folder, "train", plant_name)
            validation_folder = os.path.join(self.__output_folder, "validation", plant_name)

            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(validation_folder, exist_ok=True)

            images = self.get_all_images(os.path.join(self.__input_folder, plant_name))
            images_count = len(images)

            numpy.random.shuffle(images)

            train_images, validation_images = numpy.split(
                numpy.array(images),
                [int(images_count*0.8)]
            )

            train_images_count = len(train_images)
            validation_images_count = len(validation_images)

            print(f"""
				Data for '{plant_name}'
				Images: {images_count}
				Training: {train_images_count}
				Validation: {validation_images_count}
			""")

            # Copy-paste images
            for image_path in train_images:
                output_image = os.path.join(train_folder, os.path.basename(image_path))

                if os.path.exists(output_image):
                    output_image = os.path.join(train_folder, f"{str(random.randrange(1, 100))}_{os.path.basename(image_path)}")

                shutil.copy(image_path, output_image)
                self.make_annotations(train_annotations, output_image)

            for image_path in validation_images:
                output_image = os.path.join(validation_folder, os.path.basename(image_path))

                if os.path.exists(output_image):
                    output_image = os.path.join(validation_folder, f"{str(random.randrange(1, 100))}_{os.path.basename(image_path)}")

                shutil.copy(image_path, output_image)
                self.make_annotations(validation_annotations, output_image)
