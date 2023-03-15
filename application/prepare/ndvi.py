import imageio
import numpy
import matplotlib
import os
import random
import shutil
import rasterio
import cv2

from matplotlib import pyplot
from multiprocessing import Pool

class NDVI:
    def get_all_images(self, folder):
        images = []

        for entry in os.scandir(folder):
            if entry.is_dir():
                images.extend(self.get_all_images(entry.path))
            else:
                if entry.path.endswith(("N.tif", "N.tiff")) :
                    images.append(entry.path)

        return images

    def make_ndvi(self, input_folder, parallel=False):
        if parallel:
            self.__pool_size = 10
            self.__pool = Pool(self.__pool_size)
            self.__pool_files = []
            self.__processed_images = 0

        for plant_name in os.listdir(input_folder):
            plant_dir = os.path.join(input_folder, plant_name)

            if not os.path.isdir(plant_dir):
                continue

            ndvi_folder = os.path.join(plant_dir, "NDVI")

            shutil.rmtree(ndvi_folder, ignore_errors=True)
            os.makedirs(ndvi_folder, exist_ok=True)

            images = self.get_all_images(plant_dir)
            images_count = len(images)

            self.__pool_files = []
            self.__processed_images = 0

            for image_path in images:
                ndvi_image_path = os.path.join(ndvi_folder, os.path.basename(image_path))

                if parallel:
                    self.parallel_filter(image_path, ndvi_image_path, images_count)
                else:
                    NDVI.filter_rasterio(image_path, ndvi_image_path)

    def parallel_filter(self, input_image, output_image, total_images):
        if len(self.__pool_files) > self.__pool_size or self.__processed_images == total_images:
            self.__pool.starmap(NDVI.filter_rasterio, self.__pool_files)
            self.__pool_files = []
        self.__pool_files.append((input_image, output_image))
        self.__processed_images += 1 

    def filter_rasterio(input_image, output_image):
        print(f"Applying NDVI filter on {input_image}...")

        with rasterio.open(input_image) as _file:
            ndvi = _file.read(1)
        
        mask = cv2.threshold(ndvi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        masked_img = cv2.bitwise_and(ndvi, ndvi, mask=mask)
        ndvi = cv2.addWeighted(masked_img, 1.5, numpy.zeros_like(masked_img), 0, 0)

        cv2.imwrite(output_image, cv2.applyColorMap(ndvi, cv2.COLORMAP_SUMMER))

    def filter_rasterio_v2(input_image, output_image):
        print(f"Applying NDVI filter on {input_image}...")

        with rasterio.open(input_image) as infrared:
            infrared_data = infrared.read(1)
    
        file_name = input_image.split('N.tif')[0]

        if os.path.exists(f'{file_name}C.tif'):
            colored = f'{file_name}C.tif'
        elif os.path.exists(f'{file_name}C.tiff'):
            colored = f'{file_name}C.tiff'
        else:
            return

        with rasterio.open(colored) as colored:
            red_data = colored.read(1)

        infrared_data = infrared_data.astype('float64')
        red_data = red_data.astype('float64')

        ndvi = (infrared_data - red_data) / (infrared_data + red_data)

        output_image = output_image.split(".tif")[0] + ".tiff"

        pyplot.imsave(output_image, ndvi, cmap="RdYlGn")

    def filter(input_image, output_image):
        print(f"Applying NDVI filter on {input_image}...")

        def create_colormap(args):
            return matplotlib.colors.LinearSegmentedColormap.from_list(
            name='custom1',
            colors=[
                'gray',
                'blue',
                'green',
                'yellow',
                'red'
            ]
            )

        image = imageio.imread(input_image)
        ir = (image[:,:,0]).astype('float')
        r = (image[:,:,2]).astype('float')

        ndvi = numpy.zeros(r.size) 
        ndvi = numpy.true_divide(numpy.subtract(ir, r), numpy.add(ir, r))

        fig, ax = pyplot.subplots()
        image = ax.imshow(ndvi, cmap=create_colormap(matplotlib.colors))
        pyplot.axis('off')

        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        if os.path.exists(output_image):
            output_image_name = os.path.basename(output_image)
            output_image = output_image.replace(output_image_name, f"{str(random.randrange(1, 100))}_{output_image_name}")

        fig.savefig(output_image, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)
