import cv2
import os
import config
import pandas as pd

from multiprocessing import Pool

def in_csv(file_path, csv_data):
  splited_path = file_path.split("/")

  if files := csv_data.get(splited_path[-2]):
      file_name = splited_path[-1].split("C.")[0]
  
      if file_name.lstrip('0') in files:
          return True

def fitter(negative_image, color_image):

  print(f"Fitting image {color_image}")

  img1 = cv2.imread(negative_image, cv2.IMREAD_ANYCOLOR)
  img2 = cv2.imread(color_image, cv2.IMREAD_ANYCOLOR)

  cropped = img2[3:1533, 3:2045]
  resized_cropped = cv2.resize(cropped, (2048, 1536))
  cv2.imwrite(color_image, resized_cropped)

def scandir(input_folder):
    # csv_data = dict(
    #   pd.read_csv(config.CSV_FILE, dtype=object).groupby("Folder")["Name"].apply(list)
    # )
    
    pool = Pool(config.POOL_SIZE)
    pool_files = []

    for entry in os.scandir(input_folder):
        if entry.is_dir():
            scandir(entry.path)
        elif entry.path.endswith("C.tif") or entry.path.endswith("C.tiff"):
          color = entry.path

          file_name = color.split('C.tif')[0]

          if os.path.exists(f'{file_name}N.tif'):
            negative = f'{file_name}N.tif'
          else:
            negative = f'{file_name}N.tiff'

          if os.path.exists(negative):
            if len(pool_files) < config.POOL_SIZE:
              pool.starmap(fitter, pool_files)
              pool_files = []

            pool_files.append((negative, color))
          else:
              print(f"File {negative} not found.")

    pool.starmap(fitter, pool_files)

if __name__ == "__main__":
    scandir(config.CNN_FOLDER)
