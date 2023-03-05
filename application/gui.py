import config
import logging
import warnings
import builtins
import keras.utils.io_utils
import matplotlib
import matplotlib.pyplot
import os

import PySimpleGUI as gui

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from prepare.data import PrepareData
from prepare.ndvi import NDVI

from train.train import Train
from train.prediction import Prediction
from visualization.visualization import Visualization

warnings.filterwarnings('ignore')

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

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
    [gui.Multiline("", disabled=True, background_color="gray", size=(75, 20), no_scrollbar=True, key="-PREP_RESULTS-")],
  ]
  traning = [
    [
      gui.Text("DATA Folder", size=(12, 1)),
      gui.In(default_text=config.DATA_FOLDER, size=(47, 1), enable_events=True, key="-TRAIN_DATA_FOLDER-"),
      gui.FolderBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1))
    ],
    [
      gui.Text("Epochs", size=(12, 1)),
      gui.In(default_text=config.TRAINING_EPOCHS, size=(8, 1), enable_events=True, key="-TRAIN_EPOCHS-")
    ],
    [gui.Button(button_text="Run", size=(10, 1), key="-RUN_TRAINING-")],
    [gui.Multiline("", disabled=True, background_color="gray", size=(75, 20), no_scrollbar=True, key="-TRAIN_RESULTS_SAVED-")],
    [gui.Text("", background_color="gray", size=(75, 2), key="-TRAIN_RESULTS-")]
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
    [gui.Multiline("", disabled=True, background_color="gray", size=(75, 20), no_scrollbar=True, key="-PRED_RESULTS-")]
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

def draw_prediction(results):
  results = results[:12]

  images_count = len(results)

  if images_count == 0:
    return

  count = 1

  layout = [
    [gui.Canvas(key='-CANVAS-')]
  ]
  sub_window = gui.Window(
    'Prediction result',
    layout, 
    element_justification='center',
    finalize=True,
    resizable=True,
    font=("Arial", 12)
  )

  figure = matplotlib.pyplot.figure(figsize=(15, images_count*3))
  figure.subplots_adjust(top=0.9, bottom=0.10, left=0.01, right=0.99, hspace=0.2, wspace=0.5)

  for result in results:
    for image_path, image_class in result.items():
      axes = figure.add_subplot(round(images_count/4) + 1, 4, count)
      axes.axis('off')
      axes.imshow(matplotlib.image.imread(image_path), aspect="auto")
      axes.set_title(image_class, fontsize=10)
      count += 1 

  figure_canvas_agg = FigureCanvasTkAgg(figure, sub_window['-CANVAS-'].TKCanvas)
  figure_canvas_agg.draw()
  figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=True)

  sub_window.read(close=True)

def perform_long_operation(window, results_key, func, *args):
  def print_msg(message, line_break=True):
    if keras.utils.io_utils.is_interactive_logging_enabled():
      if line_break:
        event_name = f"{results_key[:-1]}_SAVED-"
        window[event_name].update(f"{window[event_name].get()}\n{message}")
      else:
        window[results_key].update(message)
    else:
      previous_message = window[results_key].get()
      window[results_key].update(f'{str(message)}\n{previous_message}')

  def print_wrapper(message, *args, **kwargs):
    previous_message = window[results_key].get()
    window[results_key].update(f'{str(message)}\n{previous_message}')

  builtins.print=print_wrapper

  if f"{results_key[:-1]}_SAVED-" in window.AllKeysDict:
    keras.utils.io_utils.print_msg=print_msg
  
  results = func(*args)

  window.write_event_value(f"{results_key[:-1]}_EVENT-", results)

def run_preparation(cnn_folder, data_folder, ndvi=False):
  if ndvi:
    NDVI().make_ndvi(cnn_folder)
  
  PrepareData(cnn_folder, data_folder).prepare_images()
  window["-PREP_RESULTS-"].update(f'Done!\n{window["-PREP_RESULTS-"].get()}')

def run_prediction(image_folder, classes_file, model_file):
  window["-PRED_RESULTS-"].update("")
  predict = Prediction()

  if not values.get("-PRED_MODEL_FILE-"):
    model_file = predict.get_best_model(f"{config.DATA_FOLDER}/output/models")

    window["-PRED_MODEL_FILE-"].update(model_file)
    values["-PRED_MODEL_FILE-"] = model_file
    window["-PRED_RESULTS-"].update(f"Loaded best model: {model_file}")

  if classes_file:
    predict = Prediction()

    classes = predict.load_classes(classes_file)
    resutls = predict.predict(image_folder, classes, model_file)
    predict.save_results(f"{os.path.dirname(image_folder)}/results.json", resutls)
    window["-PRED_RESULTS-"].update(f'Done!\n{window["-PRED_RESULTS-"].get()}')
    return resutls
  else:
    window["-PRED_RESULTS-"].update(f'Classes file not found!\n{window["-PRED_RESULTS-"].get()}')

def run_training(data_folder, traning_epochs):
  training = Train(data_folder)
  visualization = Visualization()
  
  train_generator, validation_generator = training.get_train_generator()
  training.save_classes(validation_generator, f"{data_folder}/validation_classes.json")

  if train_generator and validation_generator:
    history = training.train(train_generator, validation_generator, traning_epochs)
    visualization.save_history(history, f"{data_folder}/output")
    visualization.plot_accuracy(history, f"{data_folder}/output")

thread = None
window = gui.Window("CNN Monitoring", get_layout(), font=("Arial", 12))

while True:
    event, values = window.read()

    match event:
      case "-RUN_PREPARATION-":
        if thread and thread.is_alive():
          continue

        thread = window.start_thread(
          lambda: perform_long_operation(
            window, "-PREP_RESULTS-",
            run_preparation,
            values["-PREP_CNN_FOLDER-"],
            values["-PREP_DATA_FOLDER-"],
            values["-PREP_APPLY_NDVI-"]
          ),
          ('-THREAD-', '-THEAD ENDED-')
        )

      case "-RUN_TRAINING-":
        if thread and thread.is_alive():
          continue

        thread = window.start_thread(
          lambda: perform_long_operation(
            window, 
            "-TRAIN_RESULTS-",
            run_training,
            values["-TRAIN_DATA_FOLDER-"],
            int(values["-TRAIN_EPOCHS-"])
          ),
          ('-THREAD-', '-THEAD ENDED-')
        )
      case "-RUN_PREDICTION-":
        if thread and thread.is_alive():
          continue

        thread = window.start_thread(
          lambda: perform_long_operation(
            window, 
            "-PRED_RESULTS-", 
            run_prediction,
            values["-PRED_IMAGE_FOLDER-"],
            values["-PRED_CLASSES_FILE-"],
            values["-PRED_MODEL_FILE-"]
          ),
          ('-THREAD-', '-THEAD ENDED-')
        )
      case "-PRED_RESULTS_EVENT-":
        draw_prediction(values["-PRED_RESULTS_EVENT-"])
      case "OK":
        break
      case gui.WIN_CLOSED:
        break
window.close()