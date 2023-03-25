import config
import logging
import warnings
import builtins
import unidecode
import cv2
import keras.utils.io_utils
import matplotlib
import matplotlib.pyplot
import os

import PySimpleGUI as gui

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from prepare.data import PrepareData
from train.train import Train
from train.prediction import Prediction
from visualization.visualization import Visualization

warnings.filterwarnings('ignore')

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def get_layout() -> list:  # noqa
    preparation = [
        [
            gui.Text("CNN Folder", size=(12, 1)),
            gui.In(default_text=config.CNN_FOLDER, size=(47, 1),
                   enable_events=True, key="-PREP_CNN_FOLDER-"),
            gui.FolderBrowse(initial_folder=config.CNN_FOLDER, size=(10, 1))
        ],
        [
            gui.Text("DATA Folder: ", size=(12, 1)),
            gui.In(default_text=config.DATA_FOLDER, size=(47, 1),
                   enable_events=True, key="-PREP_DATA_FOLDER-"),
            gui.FolderBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1))
        ],
        [gui.Button(button_text="Run", size=(10, 1), key="-RUN_PREPARATION-")],
        [gui.Multiline("", disabled=True, background_color="gray", size=(
            75, 20), no_scrollbar=True, key="-PREP_RESULTS-")],
    ]
    traning = [
        [
            gui.Text("DATA Folder", size=(12, 1)),
            gui.In(default_text=config.DATA_FOLDER, size=(47, 1),
                   enable_events=True, key="-TRAIN_DATA_FOLDER-"),
            gui.FolderBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1))
        ],
        [
            gui.Text("Epochs", size=(12, 1)),
            gui.In(default_text=config.TRAINING_EPOCHS, size=(
                8, 1), enable_events=True, key="-TRAIN_EPOCHS-")
        ],
        [gui.Button(button_text="Run", size=(10, 1), key="-RUN_TRAINING-")],
        [gui.Multiline("", disabled=True, background_color="gray", size=(
            75, 20), no_scrollbar=True, key="-TRAIN_RESULTS_SAVED-")],
        [gui.Text("", background_color="gray",
                  size=(75, 2), key="-TRAIN_RESULTS-")]
    ]

    prediction = [
        [
            gui.Text("Model file: ", size=(13, 1)),
            gui.In(
                size=(46, 1),
                default_text=Prediction().get_best_model(
                    os.path.join(config.DATA_FOLDER, "output/models")
                ),
                enable_events=True, key="-PRED_MODEL_FILE-"
            ),
            gui.FileBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1))
        ],
        [
            gui.Text("Image to detect: ", size=(13, 1)),
            gui.In(size=(46, 1), enable_events=True,
                   key="-PRED_IMAGE_FOLDER-"),
            gui.FileBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1)),
        ],
        [
            gui.Text("Plant types file: ", size=(13, 1)),
            gui.In(default_text=os.path.join(config.DATA_FOLDER, "output/models/train_data_types.json"),
                   size=(46, 1), enable_events=True, key="-PRED_CLASSES_FILE-"),
            gui.FilesBrowse(initial_folder=config.DATA_FOLDER, size=(10, 1))
        ],
        [gui.Button(button_text="Run", size=(10, 1), key="-RUN_PREDICTION-")],
        [gui.Multiline("", disabled=True, background_color="gray", size=(
            75, 20), no_scrollbar=True, key="-PRED_RESULTS-")]
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


def draw_prediction(results: list, plant_types_file: str):  # noqa
    results = results[:4]

    images_count = len(results)

    if images_count == 0:
        return

    images_in_row = 7
    images_in_column = 7

    count = 1

    layout = [
        [gui.Canvas(key="-CANVAS-")]
    ]
    sub_window = gui.Window(
        "Prediction result",
        layout,
        element_justification="center",
        finalize=True,
        resizable=True,
        font=("Arial", 12)
    )

    plant_types = Prediction().load_classes(plant_types_file)

    figure = matplotlib.pyplot.figure(figsize=(images_in_column, images_in_row*images_count))
    figure.subplots_adjust(top=0.89, bottom=0.01, left=0.01, right=0.99, hspace=0.25, wspace=0.03)

    for result in results:
        image = cv2.imread(result[0])
        axes = figure.add_subplot(
            round(images_count/4) + 1 if images_count > 1 else 1,
            2 if images_count > 1 else 1,
            count
        )

        plants_on_image = {}
        plants_on_image_count = 0

        for data in result[1]:
            x, y, w, h = data["bbox"]

            plant_type = unidecode.unidecode(plant_types[str(data["max_index"])])
            cv2.putText(image, plant_type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            if plant_type in plants_on_image:
                plants_on_image[plant_type] += 1
            else:
                plants_on_image[plant_type] = 1

            plants_on_image_count += 1

        detected_plants = "\n".join(
            {
                f"{plant}: {round(plants_on_image[plant]*100/plants_on_image_count)}%"
                for plant in plants_on_image
            }
        )

        axes.axis("off")
        axes.set_title(f"Image: {os.path.basename(result[0])}\n{detected_plants}.", fontsize=5)
        axes.imshow(image, aspect="auto")
        count += 1

    figure_canvas_agg = FigureCanvasTkAgg(
        figure, sub_window["-CANVAS-"].TKCanvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=True)

    sub_window.read(close=True)


def perform_long_operation(window: dict, results_key: str, func: function, *args: list):  # noqa
    def print_msg(message: str, line_break: bool = True):  # noqa
        if keras.utils.io_utils.is_interactive_logging_enabled():
            if line_break:
                event_name = f"{results_key[:-1]}_SAVED-"
                window[event_name].update(
                    f"{window[event_name].get()}\n{message}")
            else:
                window[results_key].update(message)
        else:
            previous_message = window[results_key].get()
            window[results_key].update(f"{previous_message}\n{str(message)}")

    def print_wrapper(message: str, *args: list, **kwargs: dict):  # noqa
        previous_message = window[results_key].get()
        window[results_key].update(f"{previous_message}\n{str(message)}\n{''.join(args)}")

    builtins.print = print_wrapper

    if f"{results_key[:-1]}_SAVED-" in window.AllKeysDict:
        keras.utils.io_utils.print_msg = print_msg

    results = func(*args)

    window.write_event_value(f"{results_key[:-1]}_EVENT-", results)


def run_preparation(cnn_folder: str, data_folder: str):  # noqa
    PrepareData(cnn_folder, data_folder).prepare_images()
    window["-PREP_RESULTS-"].update(f'Done!\n{window["-PREP_RESULTS-"].get()}')


def run_prediction(image_folder: str, model_file: str) -> list:  # noqa
    resutls = []
    window["-PRED_RESULTS-"].update("")
    predict = Prediction()

    models_path = os.path.join(config.DATA_FOLDER, "output/models")

    if not values.get("-PRED_MODEL_FILE-"):
        model_file = predict.get_best_model(models_path)

        window["-PRED_MODEL_FILE-"].update(model_file)
        values["-PRED_MODEL_FILE-"] = model_file
        window["-PRED_RESULTS-"].update(f"Loaded best model: {model_file}")

    predict = Prediction()
    resutls = predict.predict(image_folder, model_file)

    window["-PRED_RESULTS-"].update(
        f'Done!\n{window["-PRED_RESULTS-"].get()}')

    return resutls


def run_training(data_folder: str, traning_epochs: int):  # noqa
    training = Train(data_folder)
    visualization = Visualization()

    output_folder = os.path.join(data_folder, "output")

    train_generator = training.get_data_generator("train")
    validation_generator = training.get_data_generator("validation")

    if train_generator and validation_generator:
        history = training.train(train_generator, validation_generator, traning_epochs)
        visualization.save_history(history, output_folder)
        visualization.save_traning_plot(history, output_folder)


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
                    values["-PREP_DATA_FOLDER-"]
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
                    values["-PRED_MODEL_FILE-"]
                ),
                ('-THREAD-', '-THEAD ENDED-')
            )
        case "-PRED_RESULTS_EVENT-":
            draw_prediction(
                values["-PRED_RESULTS_EVENT-"],
                values["-PRED_CLASSES_FILE-"]
            )
        case "OK":
            break
        case gui.WIN_CLOSED:
            break
window.close()
