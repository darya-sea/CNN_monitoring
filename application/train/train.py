import os
import json
import tensorflow

from keras import Model
from keras.layers import Dense, Flatten

class Train:
    def __init__(self, data_folder):
        self.__data_folder = data_folder
        self.__batch_size = 16
        self.__taget_size = (224, 224)
        self.__fix_gpu()

    def __fix_gpu(self):
        config = tensorflow.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tensorflow.compat.v1.InteractiveSession(config=config)

    def __folder_exists(self, folder):
        if not os.path.exists(folder):
            print(f"[ERROR] Data folder {folder} doesn't exist")
            return False
        return True

    def get_train_generator(self):
        train_generator, validation_generator = None, None

        datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0/255.,
            shear_range=0.2,
            zoom_range=0.2
        )
        
        train_folder = os.path.join(self.__data_folder, "train")
        validation_folder = os.path.join(self.__data_folder, "validation")

        if self.__folder_exists(train_folder):
            train_generator = datagen.flow_from_directory(
                train_folder,
                batch_size=self.__batch_size,
                shuffle=True,
                class_mode="categorical",
                target_size=self.__taget_size
            )

        if self.__folder_exists(validation_folder):
            validation_generator = datagen.flow_from_directory(
                validation_folder,
                batch_size=self.__batch_size,
                shuffle=True,
                class_mode="categorical",
                target_size=self.__taget_size
            )
        return train_generator, validation_generator

    def save_classes(self, validation_generator, classes_file):
        with open(classes_file, "w") as _file:
            _file.write(json.dumps(
                {v: k for k, v in validation_generator.class_indices.items()}))


    def train(self, train_generator, validation_generator, epochs):
        output_folder = os.path.join(self.__data_folder, "output/models")
        backup_folder = os.path.join(self.__data_folder, "backup")

        vgg_model = tensorflow.keras.applications.vgg16.VGG16(
            pooling="avg",
            weights="imagenet",
            include_top=False,
            input_shape=self.__taget_size + (3,)
        )

        for layers in vgg_model.layers:
            layers.trainable = False

        last_output = vgg_model.layers[-1].output

        vgg_x = Flatten()(last_output)
        vgg_x = Dense(128, activation="relu")(vgg_x)
        vgg_x = Dense(train_generator.num_classes, activation="softmax")(vgg_x)

        vgg_final_model = Model(vgg_model.input, vgg_x)
        vgg_final_model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(backup_folder, exist_ok=True)

        filepath = output_folder + \
            "/vgg-model-{epoch:02d}-acc-{val_acc:.2f}.hdf5"

        backup_restore = tensorflow.keras.callbacks.BackupAndRestore(
            backup_dir=backup_folder)
        checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
            filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
        early_stopping = tensorflow.keras.callbacks.EarlyStopping(
            monitor="loss", patience=10)

        history = vgg_final_model.fit(
            train_generator,
            epochs = epochs,
            validation_data = validation_generator,
            callbacks=[
                checkpoint,
                early_stopping,
                backup_restore
            ],
            verbose=1
        )

        return history
