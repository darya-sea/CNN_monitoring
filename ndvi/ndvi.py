import imageio
import numpy
import matplotlib

from matplotlib import pyplot
from multiprocessing import Pool

class NDVI:
  def __init__(self, pool_size=20):
    self.__pool = Pool(pool_size)
    self.__pool_files = []
    self.__pool_size = pool_size
    self.__processed_images = 0

  def parallel_fiiter(self, input_image, output_image, total_images):
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
