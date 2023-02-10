    
import pandas
import matplotlib.pyplot

class Visualization:
	def plot_accuracy_from_history(self, dataframe):
		dataframe.plot(figsize=(5, 5))

		matplotlib.pyplot.title('Training and Validation Accuracy')
		matplotlib.pyplot.xlabel('Epoch')
		matplotlib.pyplot.ylabel('Metric')
		matplotlib.pyplot.show()
        
	def process_history(self, history, history_file_name):
		dataframe = pandas.DataFrame(history.history)

		self.save_history(dataframe, history_file_name)
		self.plot_accuracy_from_history(dataframe)

	def save_history(self, dataframe, model_name):
		# save to json:  
		hist_json_file = model_name+'_history.json' 
		with open(hist_json_file, mode='w') as f:
			dataframe.to_json(f)

		# or save to csv: 
		hist_csv_file = model_name+'_history.csv'
		with open(hist_csv_file, mode='w') as f:
			dataframe.to_csv(f)