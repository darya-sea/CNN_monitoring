import shutil
import os
import numpy
import random
import cv2


class PrepareData:
    def __init__(self, input_folder, output_folder):
        self.__input_folder = input_folder
        self.__output_folder = output_folder

    def remove_background(self, image: cv2.Mat) -> cv2.Mat:
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

    def get_all_images(self, folder: str) -> list:
        images = []

        for entry in os.scandir(folder):
            if entry.is_dir():
                images.extend(self.get_all_images(entry.path))
            else:
                if entry.path.endswith(("C.tiff", "C.tif")):
                    images.append(entry.path)

        return images

    def make_cropped_images(self, input_image: str, output_image: str) -> int:
        images_count = 0

        image = cv2.imread(input_image)
        image = self.remove_background(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        for contour in cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            (x, y, w, h) = cv2.boundingRect(contour)
            if (w > 30 and h > 30) and (w < 150 and h < 150):
                cv2.imwrite(
                    output_image.replace(".tif", f"_{images_count}.tif"),
                    image[y:y+h, x:x+w]
                )
                images_count+=1

        return images_count

    def prepare_images(self) -> bool:
        if self.__input_folder == self.__output_folder:
            print(
                f"[ERROR] Input and output folders are same: {self.__input_folder}. Stopped.")
            return False

        train_annotations = os.path.join(self.__output_folder, "train_annotations.csv")
        validation_annotations = os.path.join(self.__output_folder, "validation_annotations.csv")

        if os.path.exists(train_annotations):
            os.remove(train_annotations)
        
        if os.path.exists(validation_annotations):
            os.remove(validation_annotations)

        for plant_name in os.listdir(self.__input_folder):
            if not os.path.isdir(os.path.join(self.__input_folder, plant_name)):
                continue

            train_folder = os.path.join(self.__output_folder, "train", plant_name)
            validation_folder = os.path.join(self.__output_folder, "validation", plant_name)

            shutil.rmtree(train_folder, ignore_errors=True)
            shutil.rmtree(validation_folder, ignore_errors=True)

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

            print(f"Preparing data for '{plant_name}'...")

            # Copy-paste images
            for image_path in train_images:
                output_image = os.path.join(train_folder, os.path.basename(image_path))

                if os.path.exists(output_image):
                    output_image = os.path.join(
                        train_folder,
                        f"{str(random.randrange(1, 100))}_{os.path.basename(image_path)}"
                    )

                #shutil.copy(image_path, output_image)
                train_images_count += self.make_cropped_images(image_path, output_image)

            for image_path in validation_images:
                output_image = os.path.join(validation_folder, os.path.basename(image_path))

                if os.path.exists(output_image):
                    output_image = os.path.join(
                        validation_folder,
                        f"{str(random.randrange(1, 100))}_{os.path.basename(image_path)}"
                    )

                #shutil.copy(image_path, output_image)
                validation_images_count += self.make_cropped_images(image_path, output_image)
            
            print(
                f"Data for '{plant_name}':\n",
                f"Input images: {images_count}\n",
				f"Training: {train_images_count}\n",
				f"Validation: {validation_images_count}\n"
			)
        return True
