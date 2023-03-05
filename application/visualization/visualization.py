    
import pandas
import os
import matplotlib.pyplot

class Visualization:
	def plot_accuracy(self, history, path, show_plot=False):
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
		dataframe = pandas.DataFrame(history.history)

		os.makedirs(path, exist_ok=True)

		hist_json_file = f"{path}/model_history.json"
		with open(hist_json_file, mode="w") as _file:
			dataframe.to_json(_file)

		hist_csv_file =  f"{path}/model_history.csv"
		with open(hist_csv_file, mode="w") as _file:
			dataframe.to_csv(_file)