import config
import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

import PySimpleGUI as gui
import matplotlib

from prepare.data import PrepareData
from prepare.ndvi import NDVI

from train.train import Train
from train.prediction import Prediction
from visualization.visualization import Visualization
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def get_layout():
  preparation = [
    [
      gui.Text("CNN Folder", size=(12, 1)),
      gui.In(default_text=config.CNN_FOLDER, size=(47, 1), enable_events=True, key="-PREP_CNN_FOLDER-"),
      gui.FolderBrowse(initial_folder=config.CNN_FOLDER, size=(10, 1))
    ],
    [
      gui.Text("DATA Folder: ", size=(12, 1)),
      gui.In(default_text=config.DATA_FOLDER, size=(47, 1), enable_events=True, key="-PREP_DATA_FOLDER-"),
      gui.FolderBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1))
    ],
    [gui.CBox("Apply NDVI", size=(12, 1), key="-PREP_APPLY_NDVI-")],
    [gui.Button(button_text="Run", size=(10,1), key="-RUN_PREPARATION-")],
    [gui.Multiline("", disabled=True, background_color="gray", size=(700, 493), key="-PREP_RESULTS-")],
  ]
  traning = [
    [
      gui.Text("DATA Folder", size=(12, 1)),
      gui.In(default_text=config.DATA_FOLDER, size=(47, 1), enable_events=True, key="-TRAIN_DATA_FOLDER-"),
      gui.FolderBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1))
    ],
    [
      gui.Text("Epochs", size=(12, 1)),
      gui.In(default_text=config.TRAINING_EPOCHS, size=(8, 1), enable_events=True, key="-TRAIN_DATA_FOLDER-")
    ],
    [gui.Button(button_text="Run", size=(10, 1), key="-RUN_TRAINING-")],
    [gui.Multiline("", disabled=True, background_color="gray", size=(700, 494), key="-TRAIN_RESULTS-")]
  ]

  prediction = [
    [
      gui.Text("Model file: ", size=(13, 1)),
      gui.In(size=(46, 1), enable_events=True, key="-PRED_MODEL_FILE-"),
      gui.FileBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1))
    ],
    [
      gui.Text("Images to detect: ", size=(13, 1)),
      gui.In(size=(46, 1), enable_events=True, key="-PRED_IMAGE_FOLDER-"),
      gui.FolderBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1)),
    ],
    [
      gui.Text("Classes file: ", size=(13, 1)),
      gui.In(default_text=f"{config.DATA_FOLDER}/validation_classes.json", size=(46, 1), enable_events=True, key="-PRED_CLASSES_FILE-"),
      gui.FilesBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1))
    ],
    [gui.Button(button_text="Run", size=(10, 1), key="-RUN_PREDICTION-")],
    [gui.Multiline("", disabled=True, background_color="gray", size=(700, 15), key="-PRED_RESULTS-")],
    [
      gui.Column(
        [[
          gui.Frame(
            "",
            [[
              gui.Canvas(key='-PREP_CANVAS-', expand_x=True, expand_y=True)
            ]],
            size=(700, 776)
          )
        ]], 
        scrollable=True, 
        vertical_scroll_only=True
      )
    ]
  ]

  layout = [
    [
      gui.TabGroup(
        [
          [gui.Tab("Preparation", preparation)],
          [gui.Tab("Traning", traning)],
          [gui.Tab("Prediction", prediction)]
        ]
      )
    ]
  ]
  return layout

def perform_long_operation(window, results_key, func, *args):
  def print_wrapper(message):
    window[results_key].update(f'{window[results_key].get()}\n{str(message)}')

  import builtins
  builtins.print=print_wrapper

  results = func(*args)

  window.write_event_value(f"{results_key[:-1]}_EVENT-", results)
  window[results_key].update(f"{window[results_key].get()}\nDone!")

window = gui.Window("CNN Monitoring", get_layout(), size=(700, 800), font=("Arial", 12))

while True:
    event, values = window.read()

    match event:
      case "-RUN_PREPARATION-":
        if values["-PREP_APPLY_NDVI-"]:
          window.start_thread(
            lambda: perform_long_operation(
              window, "-PREP_RESULTS-",
              NDVI().make_ndvi,
              values["-PREP_CNN_FOLDER-"]
            ),
            ('-THREAD-', '-THEAD ENDED-')
          )

        window.start_thread(
          lambda: perform_long_operation(
            window, 
            "-PREP_RESULTS-", 
            PrepareData(values["-PREP_CNN_FOLDER-"], values["-PREP_DATA_FOLDER-"]).prepare_images
          ),
          ('-THREAD-', '-THEAD ENDED-')
        )
      case "-RUN_TRAINING-":
        training = Train(values["-TRAIN_DATA_FOLDER-"])
        train_generator, validation_generator = training.get_train_generator()

        training.save_classes(
          validation_generator,
          f'{values["-TRAIN_DATA_FOLDER-"]}/validation_classes.json'
        )

        if train_generator and validation_generator:
          window.start_thread(
            lambda: perform_long_operation(
              window, 
              "-TRAIN_RESULTS-", 
              training.train,
              train_generator,
              validation_generator,
              config.TRAINING_EPOCHS
            ),
            ('-THREAD-', '-THEAD ENDED-')
          )
      case "-RUN_PREDICTION-":
        window["-PRED_RESULTS-"].update("")
        predict = Prediction()

        if not values.get("-PRED_MODEL_FILE-"):
          model_file = predict.get_best_model(f"{config.DATA_FOLDER}/output/models")

          window["-PRED_MODEL_FILE-"].update(model_file)
          values["-PRED_MODEL_FILE-"] = model_file
          window["-PRED_RESULTS-"].update(f"Loaded best model: {model_file}")

        if (classes_file := values.get("-PRED_CLASSES_FILE-")):
            predict = Prediction()
            classes = predict.load_classes(classes_file)
            window.start_thread(
              lambda: perform_long_operation(
                window, 
                "-PRED_RESULTS-", 
                predict.predict,
                values["-PRED_IMAGE_FOLDER-"],
                classes,
                values["-PRED_MODEL_FILE-"]
              ),
              ('-THREAD-', '-THEAD ENDED-')
            )
        else:
          window["-PRED_RESULTS-"].update(f'{window["-PRED_RESULTS-"].get()}\nClasses file not found!')
      case "-PRED_RESULTS_EVENT-":
        images_count = len(values["-PRED_RESULTS_EVENT-"])
        count = 1

        figure = matplotlib.figure.Figure(figsize=(4, 4))
  
        for result in values["-PRED_RESULTS_EVENT-"]:
          for image_path, image_class in result.items():
            axes = figure.add_subplot(round(images_count/2) + 1, 2, count)
            axes.axis('off')
            axes.imshow(matplotlib.image.imread(image_path))
            axes.set_title(image_class, fontsize=10)
            count += 1 

        figure_canvas_agg = FigureCanvasTkAgg(figure, window['-PREP_CANVAS-'].TKCanvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
      case "OK":
        break
      case gui.WIN_CLOSED:
        break
window.close()