    
import pandas
import os
import numpy
import cv2
import imutils
import matplotlib
import matplotlib.pyplot

class Visualization:
    def plot_accuracy(self, history, path, show_plot=False):
        os.makedirs(path, exist_ok=True)

        dataframe = pandas.DataFrame(history.history)
        dataframe.plot(figsize=(5, 5))

        matplotlib.pyplot.title("Training and Validation Accuracy")
        matplotlib.pyplot.xlabel("Epoch")
        matplotlib.pyplot.ylabel("Metric")
        matplotlib.pyplot.savefig(f"{path}/model_history.png")
        
        if show_plot:
            matplotlib.pyplot.show(path)
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
            
    def show_predicted_images(self, results):
        results = results[:12]
        images = []

        for result in results:
            image = cv2.imread(result[0])
            (h, w) = image.shape[:2]

            startX = int(result[1] * w)
            startY = int(result[2] * h)
            endX = int(result[3] * w)
            endY = int(result[4] * h)

            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.putText(image, os.path.basename(result[0]), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "TEST", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            images.append(image)

        cv2.imshow("Prediction result", numpy.concatenate(images, axis=1))
        cv2.waitKey(0)
        cv2.destroyAllWindows()