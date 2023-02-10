import sys
import config
import logging
import warnings

warnings.filterwarnings('ignore')

from fitter.fitter import Fitter

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

def prepare():
  from prepare.data import PrepareData
  from prepare.ndvi import NDVI

  NDVI().make_ndvi(config.CNN_FOLDER, parallel=True)
  PrepareData(config.CNN_FOLDER, config.DATA_FOLDER).prepare_images()

def predict(image_path):
  from training.prediction import Prediction

  prediction = Prediction(config.DATA_FOLDER)
  print(prediction.predict(image_path))

def train():
  from training.training import Training
  from visualization.visualization import Visualization

  training = Training(config.DATA_FOLDER)
  train_generator, validation_generator = training.get_train_generator()
  if train_generator and validation_generator:
      visualization = Visualization()
      # training.validation(train_generator, validation_generator, config.TRAINING_EPOCHS)
      history = training.train(train_generator, validation_generator, config.TRAINING_EPOCHS)
      visualization.process_history(history, f"{config.DATA_FOLDER}/vgg_model")

if __name__ == "__main__":
  if len(sys.argv) > 1:
    match sys.argv[1]:
      case "help":
        print(f"usage: {sys.argv[1]} <prepare|train|predict>")
      case "prepare":
        prepare()
      case "train":
        train()
      case "predict":
        predict(sys.argv[2])