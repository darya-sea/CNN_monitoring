import shutil
import os
import numpy
import random
import cv2


class PrepareData:
    def __init__(self, input_folder, output_folder):
        self.__input_folder = input_folder
        self.__output_folder = output_folder

    def backround_remove(self, image_path):
        image = cv2.imread(image_path)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_chlorophyll = numpy.array([40, 50, 50])
        upper_chlorophyll = numpy.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower_chlorophyll, upper_chlorophyll)

        kernel = numpy.ones((5,5), numpy.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        res = cv2.bitwise_and(image, image, mask=mask)
        bg = numpy.zeros_like(image)
        bg[mask != 0] = res[mask != 0]
        cv2.imwrite(image_path, bg)

    def get_all_images(self, folder):
        images = []

        for entry in os.scandir(folder):
            if entry.is_dir():
                images.extend(self.get_all_images(entry.path))
            else:
                if entry.path.endswith(("N.tiff", "N.tif")):
                    images.append(entry.path)

        return images

    def save_annotations(self, csv_file, image_path, output_image):
        plant_name = os.path.basename(os.path.dirname(output_image))
        
        image_name = os.path.basename(output_image).replace("C.", "N.").split("_")[-1]

        ndvi_folder = os.path.join(os.path.dirname(os.path.dirname(image_path)), "NDVI")
        ndvi_image = os.path.join(ndvi_folder, image_name)

        if os.path.exists(ndvi_image):
            image = cv2.imread(ndvi_image, cv2.IMREAD_GRAYSCALE)
            mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            with open(csv_file, "a") as _file:
                for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if w > 40:
                        _file.write(f"{output_image},{x},{y},{w},{h},{plant_name}\n")

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
                self.save_annotations(train_annotations, image_path, output_image)

            for image_path in validation_images:
                output_image = os.path.join(validation_folder, os.path.basename(image_path))

                if os.path.exists(output_image):
                    output_image = os.path.join(validation_folder, f"{str(random.randrange(1, 100))}_{os.path.basename(image_path)}")

                shutil.copy(image_path, output_image)
                self.save_annotations(validation_annotations, image_path, output_image)
