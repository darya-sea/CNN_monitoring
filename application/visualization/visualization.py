    
import pandas
import os
import cv2
import matplotlib
import matplotlib.pyplot
import unidecode

class Visualization:
    def save_traning_plot(self, history, path):
        os.makedirs(path, exist_ok=True)

        dataframe = pandas.DataFrame(history.history)
        dataframe.plot(figsize=(5, 5))

        matplotlib.pyplot.title("Training and Validation Accuracy")
        matplotlib.pyplot.xlabel("Epoch")
        matplotlib.pyplot.ylabel("Metric")
        matplotlib.pyplot.savefig(f"{path}/model_history.png")
        matplotlib.pyplot.close()
    
    def show_traning_plot(self, json_file):
        if os.path.exists(json_file):
            dataframe = pandas.read_json(json_file)

            plots = [
                {
                    "filter": ("loss", "acc"),
                    "title": "Training and Validation Accuracy",
                    "figure_path": "model_history.png"
                }
            ]

            for plot in plots:
                plot_dataframe = dataframe.drop(
                    columns=[
                        column
                        for column in dataframe.columns
                        if column not in plot["filter"]
                    ]
                )
                plot_dataframe.plot(figsize=(5, 5))

                matplotlib.pyplot.title(f"Training and Validation Accuracy")
                matplotlib.pyplot.xlabel("Epoch")
                matplotlib.pyplot.ylabel("Metric")
                matplotlib.pyplot.savefig(os.path.join(os.path.dirname(json_file), plot["figure_path"]))
                matplotlib.pyplot.show()
                matplotlib.pyplot.close()


    def save_history(self, history, path):
        os.makedirs(path, exist_ok=True)

        dataframe = pandas.DataFrame(history.history)

        hist_json_file = f"{path}/model_history.json"
        with open(hist_json_file, mode="w") as _file:
            dataframe.to_json(_file)

        hist_csv_file =  f"{path}/model_history.csv"
        with open(hist_csv_file, mode="w") as _file:
            dataframe.to_csv(_file)
            
    def show_predicted_images(self, results, plant_types):
        results = results[:12]
        images_count = len(results)
        count = 1

        images_in_row = 3
        images_in_column = 15

        figure = matplotlib.pyplot.figure(figsize=(images_in_column, images_count*images_in_row))
        figure.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99, hspace=0.11, wspace=0.03)

        for result in results:
            image = cv2.imread(result[0])
            axes = figure.add_subplot(images_count, 1, count)    

            for data in result[1]:
                x, y, w, h = data["bbox"]

                plant_type = unidecode.unidecode(plant_types[str(data["max_index"])])
                cv2.putText(image, plant_type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            axes.axis('off')
            axes.set_title(f"{os.path.basename(result[0])}", fontsize=9)
            axes.imshow(image, aspect="auto")
            count += 1

        matplotlib.pyplot.show()