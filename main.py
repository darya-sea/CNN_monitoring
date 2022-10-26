import cv2
import os
import config

from multiprocessing import Pool

def fitter(negative_image, color_image):
    print(f"Fitting image {color_image}")

    img1 = cv2.imread(negative_image, cv2.IMREAD_ANYCOLOR)
    img2 = cv2.imread(color_image, cv2.IMREAD_ANYCOLOR)

    cropped = img2[3:1533, 3:2045]
    resized_cropped = cv2.resize(cropped, (2048, 1536))
   
    cv2.imwrite(color_image, resized_cropped)

def scandir(input_folder):
    pool = Pool(config.POOL_SIZE)
    pool_files = []

    for entry in os.scandir(input_folder):
        if entry.is_dir():
            scandir(entry.path)
        elif entry.path.endswith("C.tif") or entry.path.endswith("C.tiff"):
            color = entry.path
            negative = color.replace('C.tif', 'N.tif')

            if os.path.exists(negative):

                if len(pool_files) < config.POOL_SIZE:
                    pool_files.append((negative, color))
                else:
                    pool.starmap(fitter, pool_files)
                    pool_files = []
            else:
                print(f"File {negative} not found.")

    pool.starmap(fitter, pool_files)

if __name__ == "__main__":
    scandir(config.CNN_FOLDER)