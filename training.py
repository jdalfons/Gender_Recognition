from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model.convNet import CnnVgg
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import matplotlib


class Training:

    def __init__(self, CATEGORIES, DATASET, DATADIR):

        self.CATEGORIES = CATEGORIES
        self.DATASET = DATASET
        self.DATADIR = DATADIR
        # Constant to render the images
        self.epochs = 100
        self.lr = 1e-3
        self.batch_size = 64
        self.img_dims = (96, 96, 3)
        self.plotName = "./img/graf.png"
        self.model = "./training/gender_detection.model"

    def create_training_data(self):

        matplotlib.use("Agg")
        files = [f for f in glob.glob(str(self.DATADIR) + "/**/*",
                 recursive=True)if not os.path.isdir(f)]
        random.seed(42)
        random.shuffle(files)

        # Iterating each image, read and resize with Opencv.
        for img in files:
            try:
                image = cv2.imread(img)

                image = cv2.resize(image, (self.img_dims[0],
                                           self.img_dims[1]))
                image = img_to_array(image)
                self.DATASET.append(image)

                label = img.split(os.path.sep)[-2]
                if label == "woman":
                    label = 1
                else:
                    label = 0
                self.CATEGORIES.append([label])

            # Exeptions
            except Exception as e:
                pass
            except OSError as e:
                print("OSErrroBad img:", e, os.path.join(self.DATADIR, img))
            except Exception as e:
                print("general exception:", e, os.path.join(self.DATADIR, img))

        # Pre-processing
        self.DATASET = np.array(self.DATASET, dtype="float") / 255.0
        self.CATEGORIES = np.array(self.CATEGORIES)

        # split dataset for training and validation
        (trainX, testX, trainY, testY) = train_test_split(self.DATASET,
                                                          self.CATEGORIES,
                                                          test_size=0.2,
                                                          random_state=42)
        trainY = to_categorical(trainY, num_classes=2)
        testY = to_categorical(testY, num_classes=2)

        # augmenting datset
        aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True,
                                 fill_mode="nearest")

        # build model
        model = CnnVgg.build(width=self.img_dims[0],
                             height=self.img_dims[1],
                             depth=self.img_dims[2],
                             classes=2)

        # compile the model
        opt = Adam(lr=self.lr, decay=self.lr/self.epochs)
        model.compile(loss="binary_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])

        # train the model
        H = model.fit_generator(aug.flow
                                (trainX, trainY, batch_size=self.batch_size),
                                validation_data=(testX, testY),
                                steps_per_epoch=len(trainX) // self.batch_size,
                                epochs=self.epochs, verbose=1)

        # save the model to disk
        model.save(self.model)

        # plot training/validation loss/accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = self.epochs
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")

        # saving plot to disk
        plt.savefig(self.plotName)


if __name__ == "__main__":

    # Variables to pass in the class
    datadir = "./gender_dataset_face/"
    dataset = []
    categories = []

    # Object
    training_model = Training(categories, dataset, datadir)

    # Proces to start the training
    training_model.create_training_data()
