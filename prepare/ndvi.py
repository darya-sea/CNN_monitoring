import imageio
import numpy
import matplotlib
import os
import random

from matplotlib import pyplot
from multiprocessing import Pool

class NDVI:
  def __init__(self):
    self.__pool_size = 10
    self.__pool = Pool(self.__pool_size)
    self.__pool_files = []
    self.__processed_images = 0

  def get_all_images(self, folder):
    images = []

    for entry in os.scandir(folder):
        if entry.is_dir():
          images.extend(self.get_all_images(entry.path))
        else:
          if ".DS_Store" not in entry.path and "N.tif" not in entry.path:
            images.append(entry.path)

    return images

  def make_ndvi(self, input_folder, parallel=False):
    all_classes = os.listdir(input_folder)

    for _class in all_classes:
      _class_dir = f"{input_folder}/{_class}"

      if not os.path.isdir(_class_dir):
        continue

      ndvi_folder = f"{_class_dir}/NDVI"
      os.makedirs(ndvi_folder, exist_ok=True)

      images = self.get_all_images(_class_dir)
      images_count = len(images)

      self.__pool_files = []
      self.__processed_images = 0

      for image_path in images:
        output_image = f"{ndvi_folder}/{os.path.basename(image_path)}"
    
        if os.path.exists(output_image):
          output_image = f"{ndvi_folder}/{str(random.randrange(1, 100))}_{os.path.basename(image_path)}"

        if parallel:
          self.parallel_filter(image_path, output_image, images_count)
        else:
          self.filter(image_path, output_image)

  def parallel_filter(self, input_image, output_image, total_images):
    if len(self.__pool_files) > self.__pool_size or self.__processed_images == total_images:
      self.__pool.starmap(NDVI.filter, self.__pool_files)
      self.__pool_files = []
    self.__pool_files.append((input_image, output_image))
    self.__processed_images += 1 

  @staticmethod
  def filter(input_image, output_image):
    print(f"Applying NDVI filter on {output_image}...")

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

    fig.savefig(output_image, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)
