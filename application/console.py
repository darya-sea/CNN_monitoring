import sys
import config
import logging
import warnings
import os

from pprint import pprint

warnings.filterwarnings('ignore')

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def prepare(option=None):
    from prepare.data import PrepareData
    from prepare.ndvi import NDVI

    if option == "ndvi" or option == None:
        NDVI().make_ndvi(config.CNN_FOLDER, parallel=True)

    if option == "data" or option == None:
        PrepareData(config.CNN_FOLDER, config.DATA_FOLDER).prepare_images()


def predict(image_path):
    from train.prediction import Prediction
    from visualization.visualization import Visualization

    predict = Prediction()
    visualization = Visualization()

    models_path = os.path.join(config.DATA_FOLDER, "output/models")
    classes_path = os.path.join(config.DATA_FOLDER, "validation_classes.json")

    model_file = predict.get_best_model(models_path)

    if model_file:
        classes = predict.load_classes(classes_path)
        resutls = predict.predict(image_path, classes, model_file)
        if resutls:
            visualization.show_predicted_images(resutls)
            pprint(resutls)

def train():
    from train.train import Train
    from visualization.visualization import Visualization

    training = Train(config.DATA_FOLDER)
    visualization = Visualization()
    
    classes_path = os.path.join(config.DATA_FOLDER, "validation_classes.json")
    history_path = os.path.join(config.DATA_FOLDER, "output")

    train_generator, validation_generator = training.get_train_generator()
    if train_generator and validation_generator:
        training.save_classes(validation_generator, classes_path)
        history = training.train(train_generator, validation_generator, config.TRAINING_EPOCHS)
        visualization.plot_accuracy(history, history_path)
        visualization.save_history(history, history_path)

def help(script_name):
    print(
    f"""
        usage: {script_name} <prepare|train|predict>

        preapre example: 
          python {script_name} prepare ndvi
          python {script_name} prepare data
          python {script_name} prepare 
        train example: 
          python {script_name} train
        predict example: 
          python {script_name} predict "CNN/heřmánkovec nevonný/2022_09_21 hermankovec/00257C.tif"
    """
    )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "help":
                help(sys.argv[0])
            case "prepare":
                prepare(sys.argv[2] if len(sys.argv) > 2 else None)
            case "train":
                train()
            case "predict":
                if len(sys.argv) > 2:
                    predict(sys.argv[2])
                else:
                    print(f"usage: {sys.argv[0]} predict <image_path>")
    else:
        help(sys.argv[0])
