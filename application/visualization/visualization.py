    
import pandas
import os

from matplotlib import image
from matplotlib import pyplot

class Visualization:
    def plot_accuracy(self, history, path, show_plot=False):
        os.makedirs(path, exist_ok=True)

        dataframe = pandas.DataFrame(history.history)
        dataframe.plot(figsize=(5, 5))

        pyplot.title("Training and Validation Accuracy")
        pyplot.xlabel("Epoch")
        pyplot.ylabel("Metric")
        pyplot.savefig(f"{path}/model_history.png")
        
        if show_plot:
            pyplot.show(path)
        pyplot.close()

    def save_history(self, history, path):
        os.makedirs(path, exist_ok=True)

        dataframe = pandas.DataFrame(history.history)

        hist_json_file = f"{path}/model_history.json"
        with open(hist_json_file, mode="w") as _file:
            dataframe.to_json(_file)

        hist_csv_file =  f"{path}/model_history.csv"
        with open(hist_csv_file, mode="w") as _file:
            dataframe.to_csv(_file)
            
    def show_predicted_images(self, results):
        results = results[:12]

        images_count = len(results)
        count = 1

        figure = pyplot.figure(figsize=(15, images_count*3))

        for result in results:
            for image_path, image_class in result.items():
                axes = figure.add_subplot(round(images_count/4) + 1, 4, count)
                axes.axis('off')
                axes.imshow(image.imread(image_path), aspect="auto")
                axes.set_title(image_class, fontsize=10)
                count += 1

        pyplot.show()
