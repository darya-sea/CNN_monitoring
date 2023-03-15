import os
import json
import tensorflow
import numpy

from keras import Model, utils
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import image_utils as keras_image_utils
from sklearn.preprocessing import LabelBinarizer


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
    
    def get_train_data(self):
        label_binarizer = LabelBinarizer()

        train_images = []
        train_labels = []
        train_bboxes = []
        train_paths = []

        validation_images = []
        validation_labels = []
        validation_bboxes = []
        validation_paths = []

        train_annotations = os.path.join(self.__data_folder, "train_annotations.csv")
        validation_annotations = os.path.join(self.__data_folder, "validation_annotations.csv")

        if os.path.exists(train_annotations):
            with open(train_annotations, "r") as _file:
                for line in _file.readlines():
                    annotations = line.split(",")
                    image = keras_image_utils.load_img(annotations[0], target_size=self.__taget_size)
                    h, w = image.size[0],image.size[1]

                    train_images.append(keras_image_utils.img_to_array(image))
                    train_paths.append(annotations[0])
                    train_bboxes.append( 
                        (
                            float(annotations[1])/w,
                            float(annotations[2])/h,
                            float(annotations[3])/w,
                            float(annotations[4])/h
                        )
                    )
                    train_labels.append(annotations[5])
            
            train_images = numpy.array(train_images, dtype="float32") / 255.0
            train_labels = numpy.array(train_labels)
            train_bboxes = numpy.array(train_bboxes, dtype="float32")
            train_paths = numpy.array(train_paths)

            train_labels = label_binarizer.fit_transform(train_labels)

            if len(label_binarizer.classes_) == 2:
                train_labels = utils.to_categorical(train_labels)

        if os.path.exists(validation_annotations):
            with open(validation_annotations, "r") as _file:
                for line in _file.readlines():
                    annotations = line.split(",")
                    image = keras_image_utils.load_img(annotations[0], target_size=self.__taget_size)
                    h, w = image.size[0],image.size[1]

                    validation_images.append(keras_image_utils.img_to_array(image))
                    validation_paths.append(annotations[0])
                    validation_bboxes.append(
                        (
                            float(annotations[1])/w,
                            float(annotations[2])/h,
                            float(annotations[3])/w,
                            float(annotations[4])/h
                        )
                    )
                    validation_labels.append(annotations[5])

            validation_images = numpy.array(validation_images, dtype="float32") / 255.0
            validation_labels = numpy.array(validation_labels)
            validation_bboxes = numpy.array(validation_bboxes, dtype="float32")
            validation_paths = numpy.array(validation_paths)

            validation_labels = label_binarizer.fit_transform(validation_labels)

            if len(label_binarizer.classes_) == 2:
               validation_labels = utils.to_categorical(validation_labels)

        train_targets = {
            "class_label": train_labels,
            "bounding_box": train_bboxes
        }

        validation_targets = {
            "class_label": validation_labels,
            "bounding_box": validation_bboxes
        }

        return train_images, train_targets, validation_images, validation_targets

    def save_classes(self, validation_generator, classes_file):
        with open(classes_file, "w") as _file:
            _file.write(json.dumps(
                {v: k for k, v in validation_generator.class_indices.items()}))

    def train(self, train_images, train_targets, validation_images, validation_targets, epochs):
        label_binarizer = LabelBinarizer()

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

        flatten = Flatten()(last_output)
        # vgg_x = Dense(128, activation="relu")(vgg_x)
        # vgg_x = Dense(train_generator.num_classes, activation="softmax")(vgg_x)

        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

        softmaxHead = Dense(512, activation="relu")(flatten)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(len(label_binarizer.classes_), activation="softmax", name="class_label")(softmaxHead)

        vgg_final_model = Model(vgg_model.input, outputs=(bboxHead, softmaxHead))
        vgg_final_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

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
            train_images, train_targets,
	        validation_data=(validation_images, validation_targets),
            epochs = epochs,
            callbacks=[
                checkpoint,
                early_stopping,
                backup_restore
            ],
            verbose=1
        )

        return history
