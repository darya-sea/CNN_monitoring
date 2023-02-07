import shutil
import os
import numpy

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
          if ".DS_Store" not in entry.path:
            images.append(entry.path)

    return images

  def prepare_images(self):
    if self.__input_folder == self.__output_folder:
      print(f"[ERROR] Input and output folders are same: {self.__input_folder}. Stopped.")
      return False

    all_classes = os.listdir(self.__input_folder)
    shutil.rmtree(self.__output_folder, ignore_errors=True)

    for _class in all_classes:
      train_folder = f"{self.__output_folder}/train/{_class}"
      validation_folder = f"{self.__output_folder}/validation/{_class}"

      os.makedirs(train_folder, exist_ok=True)
      os.makedirs(validation_folder, exist_ok=True)

      images = self.get_all_images(f"{self.__input_folder}/{_class}")
      images_count = len(images)

      numpy.random.shuffle(images)

      train_images, validation_images = numpy.split(
          numpy.array(images), 
          [int(images_count*0.8)]
      )

      train_images_count = len(train_images)
      validation_images_count = len(validation_images)

      # View count of images after partition
      print(f"""
          Data for '{_class}'
          Images: {images_count}
          Training: {train_images_count}
          Validation: {validation_images_count}
      """)

      # Copy-paste images
      for image_path in train_images:
        shutil.copy(image_path, train_folder)

      for image_path in validation_images:
        shutil.copy(image_path, validation_folder)
