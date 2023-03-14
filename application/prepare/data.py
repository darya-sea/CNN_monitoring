import shutil
import os
import numpy
import random


class PrepareData:
    def __init__(self, input_folder, output_folder):
        self.__input_folder = input_folder
        self.__output_folder = output_folder

    def get_all_images(self, folder):
        images = []

        for entry in os.scandir(folder):
            if entry.is_dir():
                images.extend(self.get_all_images(entry.path))
            else:
                if entry.path.endswith((".tiff", ".tif", ".png", ".jpg", ".jpeg")):
                    images.append(entry.path)

        return images

    def prepare_images(self):
        if self.__input_folder == self.__output_folder:
            print(
                f"[ERROR] Input and output folders are same: {self.__input_folder}. Stopped.")
            return False

        shutil.rmtree(self.__output_folder, ignore_errors=True)

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

            for image_path in validation_images:
                output_image = os.path.join(validation_folder, os.path.basename(image_path))

                if os.path.exists(output_image):
                    output_image = os.path.join(validation_folder, f"{str(random.randrange(1, 100))}_{os.path.basename(image_path)}")

                shutil.copy(image_path, output_image)
