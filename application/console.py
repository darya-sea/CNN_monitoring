import sys
import config
import logging
import warnings

from pprint import pprint

warnings.filterwarnings('ignore')

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

def prepare(ndvi=None):
  from prepare.data import PrepareData
  from prepare.ndvi import NDVI

  if ndvi == "ndvi" or ndvi == None:
    NDVI().make_ndvi(config.CNN_FOLDER, parallel=True)
  
  if ndvi == "data" or ndvi == None:
    PrepareData(config.CNN_FOLDER, config.DATA_FOLDER).prepare_images()

def predict(image_path):
  from train.prediction import Prediction

  predict = Prediction()

  model_file = predict.get_best_model(f"{config.DATA_FOLDER}/output/models")
  classes = predict.load_classes(f"{config.DATA_FOLDER}/validation_classes.json")

  pprint(predict.predict(image_path, classes, model_file))

def train():
  from train.train import Train
  from visualization.visualization import Visualization

  training = Train(config.DATA_FOLDER)
  train_generator, validation_generator = training.get_train_generator()
  if train_generator and validation_generator:
      training.save_classes(validation_generator, f"{config.DATA_FOLDER}/validation_classes.json")
      visualization = Visualization()
      # training.validation(train_generator, validation_generator, config.TRAINING_EPOCHS)
      history = training.train(train_generator, validation_generator, config.TRAINING_EPOCHS)
      # visualization.process_history(history, f"{config.DATA_FOLDER}/output/history")

if __name__ == "__main__":
  if len(sys.argv) > 1:
    match sys.argv[1]:
      case "help":
        print(
        f"""
        usage: {sys.argv[0]} <prepare|train|predict>

        preapre example: 
          python main.py prepare ndvi
          python main.py prepare data
          python main.py prepare 
        train example: 
          python main.py train
        predict example: 
          python main.py predict "CNN/heřmánkovec nevonný/2022_09_21 hermankovec/00257C.tif"
        """
        )
      case "prepare":
        prepare(sys.argv[2] if len(sys.argv) > 2 else None )
      case "train":
        train()
      case "predict":
        if len(sys.argv) > 2:
          predict(sys.argv[2])
        else:
          print(f"usage: {sys.argv[0]} predict <image_path>")