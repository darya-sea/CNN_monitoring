import config
import logging
import warnings

warnings.filterwarnings('ignore')

from training.training import Training
from training.prepare import PrepareImages
from visualization.visualization import Visualization
from fitter.fitter import Fitter

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

def main():
    training = Training(config.DATA_FOLDER)
    visualization = Visualization()
    #prepare = PrepareImages(config.CNN_FOLDER, config.DATA_FOLDER)

    #prepare.prepare_images()
    train_generator, validation_generator = training.get_train_generator()
    # training.validation(train_generator, validation_generator, config.TRAINING_EPOCHS)
    history = training.train(train_generator, validation_generator, config.TRAINING_EPOCHS)
    visualization.process_history(history, f"{config.DATA_FOLDER}/vgg_model")

if __name__ == "__main__":
    main()