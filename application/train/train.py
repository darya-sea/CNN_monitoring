import os
import json
import numpy
import tensorflow
import pandas

from keras import Model
from keras.layers import Dense, Flatten, Dropout


class Train:
    def __init__(self, data_folder):
        self.__data_folder = data_folder
        self.__batch_size = 16
        self.__taget_size = (224, 224)
        self.__fix_gpu()

        self.__output_folder = os.path.join(self.__data_folder, "output/models")
        self.__backup_folder = os.path.join(self.__data_folder, "backup")

        os.makedirs(self.__output_folder, exist_ok=True)
        os.makedirs(self.__backup_folder, exist_ok=True)

    def __fix_gpu(self):
        config = tensorflow.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tensorflow.compat.v1.InteractiveSession(config=config)

    def __path_exists(self, folder):
        if not os.path.exists(folder):
            print(f"[ERROR] Data folder {folder} doesn't exist")
            return False
        return True

    def get_data_flow_generator(self, csv_file, data_folder_name):
        dataframe_columns = ["image_path", "x", "y", "w", "h", "plant_name"]

        datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0/255.,
            shear_range=0.2,
            zoom_range=0.2
        )
        
        data_folder = os.path.join(self.__data_folder, data_folder_name)

        if self.__path_exists(data_folder) and self.__path_exists(csv_file):
            dataframe = pandas.read_csv(csv_file, names=dataframe_columns, header=None)

            dataframe["classes_one_hot"] = dataframe[dataframe_columns[5]].str.get_dummies().values.tolist()
            dataframe["bbox"] = dataframe[
                [
                    dataframe_columns[1],
                    dataframe_columns[2],
                    dataframe_columns[3],
                    dataframe_columns[4]
                ]
            ].values.tolist()

            generator = datagen.flow_from_dataframe(
                dataframe=dataframe,
                directory=data_folder,
                x_col=dataframe_columns[0],
                y_col=["bbox", "classes_one_hot"],
                batch_size=self.__batch_size,
                shuffle=True,
                class_mode="multi_output",
                target_size=self.__taget_size,
                seed=42
            )
            plant_types = dataframe[dataframe_columns[5]].str.get_dummies().keys().to_frame(index=False).to_dict()[0]
            json_file = os.path.join(
                self.__output_folder,
                f"{data_folder_name}_plant_types.json"
            )

            with open(json_file, "w") as _file:
                _file.write(json.dumps(plant_types))
            return generator

    def __create_generator(self, data_flow_iterator):
        while True:
            images, labels = data_flow_iterator.next()

            targets = {
                'class_label': labels[1],
                'bounding_box': labels[0]
            }
            yield images, targets

    def train(self, train_generator, validation_generator, epochs):
        vgg_model = tensorflow.keras.applications.vgg16.VGG16(
            pooling="avg",
            weights="imagenet",
            include_top=False,
            input_shape=self.__taget_size + (3,)
        )

        for layers in vgg_model.layers:
            layers.trainable = False

        last_output = vgg_model.layers[-1].output

        flatten = Flatten()(last_output)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

        softmaxHead = Dense(512, activation="relu")(flatten)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(5, activation="softmax", name="class_label")(softmaxHead)

        vgg_final_model = Model(vgg_model.input, outputs=(bboxHead, softmaxHead))
        vgg_final_model.compile(
            loss={
                "class_label": "categorical_crossentropy",
                "bounding_box": "mean_squared_error"
            }, 
            optimizer="adam", 
            metrics=["acc"],
            loss_weights={
                "class_label": 1.0,
                "bounding_box": 1.0
            }
        )

        filepath = os.path.join(self.__output_folder , "vgg-model-{epoch:02d}-acc-{class_label_acc:.2f}.hdf5")

        backup_restore = tensorflow.keras.callbacks.BackupAndRestore(backup_dir=self.__backup_folder)
        checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
            filepath, monitor="class_label_acc", verbose=1, save_best_only=True, mode="max")
        early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

        history = vgg_final_model.fit_generator(
            self.__create_generator(train_generator),
	        validation_data=self.__create_generator(validation_generator),
            steps_per_epoch=train_generator.n//self.__batch_size,
            validation_steps=validation_generator.n//self.__batch_size,
            epochs = epochs,
            callbacks=[
                checkpoint,
                early_stopping,
                backup_restore
            ],
            verbose=1
        )

        return history
