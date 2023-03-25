import pandas
import os
import cv2
import matplotlib
import matplotlib.pyplot
import unidecode


class Visualization:
    """Class to show results."""

    def save_traning_plot(self, history: any, path: str):
        """Save traning plot.

        Args:
            history (dataframe): training history dataframe.
            path (str): path to save plot.
        """
        os.makedirs(path, exist_ok=True)

        dataframe = pandas.DataFrame(history.history)
        dataframe.plot(figsize=(5, 5))

        matplotlib.pyplot.title("Training and Validation Accuracy")
        matplotlib.pyplot.xlabel("Epoch")
        matplotlib.pyplot.ylabel("Metric")
        matplotlib.pyplot.savefig(f"{path}/model_history.png")
        matplotlib.pyplot.close()

    def show_traning_plot(self, json_file: str):
        """Show traning plit.

        Args:
            json_file (str): training data to draw plot.
        """
        if os.path.exists(json_file):
            dataframe = pandas.read_json(json_file)

            plots = [
                {
                    "filter": ("loss", "acc", "val_loss", "val_acc"),
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

    def save_history(self, history: any, path: str):
        """Save traning hisotry to csv/json.

        Args:
            history (dataframe): training history dataframe.
            path (str): path to save history in csv/json format.
        """
        os.makedirs(path, exist_ok=True)

        dataframe = pandas.DataFrame(history.history)

        hist_json_file = f"{path}/model_history.json"
        with open(hist_json_file, mode="w") as _file:
            dataframe.to_json(_file)

        hist_csv_file = f"{path}/model_history.csv"
        with open(hist_csv_file, mode="w") as _file:
            dataframe.to_csv(_file)

    def show_predicted_images(self, results: list, data_types: dict):
        """Show predicted images.

        Args:
            results (list): list of predicted results.
            data_types (dict): types of data to extract by predicted index..
        """
        results = results[:4]
        images_count = len(results)
        count = 1

        images_in_row = 3
        images_in_column = 10

        figure = matplotlib.pyplot.figure(figsize=(images_in_column, images_count + images_in_row))
        figure.subplots_adjust(top=0.9, bottom=0.01, left=0.01, right=0.99, hspace=0.25, wspace=0.03)

        for result in results:
            image = cv2.imread(result[0])
            axes = figure.add_subplot(
                round(images_count/4) + 1 if images_count > 1 else 1,
                2 if images_count > 1 else 1,
                count
            )

            objects_on_image = {}
            objects_on_image_count = 0

            for data in result[1]:
                x, y, w, h = data["bbox"]

                data_type = unidecode.unidecode(data_types[str(data["max_index"])])
                cv2.putText(image, data_type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

                if data_type in objects_on_image:
                    objects_on_image[data_type] += 1
                else:
                    objects_on_image[data_type] = 1

                objects_on_image_count += 1

            detected_objects = "\n".join(
                {
                    f"{object}: {round(objects_on_image[object]*100/objects_on_image_count)}%"
                    for object in objects_on_image
                }
            )

            axes.axis("off")
            axes.set_title(f"Image: {os.path.basename(result[0])}\n{detected_objects}.", fontsize=9)
            axes.imshow(image, aspect="auto")
            count += 1

        matplotlib.pyplot.show()
