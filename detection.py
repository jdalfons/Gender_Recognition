from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import cv2
import os
import cvlib as cv
import numpy as np


def show_Cam():

    url = "https://storage.googleapis.com/pretrained-model-gender/gender_recognition.model"
    model_path = get_file("gender_detection.model",
                          url,
                          cache_subdir="training", cache_dir=os.getcwd())
    model = load_model(model_path)
    classes = ['man', 'woman']

    # Capture the video from object 0 (Webcam)
    webcam = cv2.VideoCapture(0)

    # While the camera is active
    while (True):
        try:
            # Get frames from the camera and read it
            ret, frame = webcam.read()

            # With cvlib internal training convolute to recognice a face
            face, conf = cv.detect_face(frame)
            print(face)
            print(conf)

            # Get the each frame and recognize where is faces
            for id, p in enumerate(face):

                # Get the rectangle shape of the face
                (startX, startY) = p[0], p[1]
                (endX, endY) = p[2], p[3]

                # draw lines from the rectangle conrdenates
                cv2.rectangle(frame,
                              (startX, startY),
                              (endX, endY),
                              (255, 0, 0))

                face_crop = np.copy(frame[startY:endY, startX:endX])

                resize_crop = cv2.resize(face_crop, (96, 96))
                resize_crop = resize_crop.astype("float") / 255
                resize_crop = img_to_array(resize_crop)
                resize_crop = np.expand_dims(resize_crop, axis=0)

                predict = model.predict(resize_crop)[0]
                print(predict)
                print(classes)

                id = np.argmax(predict)
                label_text = classes[id]

                label_text = "{}: {:.2f}%".format(label_text,
                                                  predict[id] * 100)
                # Calc frames og startY to write the text
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # Write in top of the rectangle
                cv2.putText(frame, label_text, (startX, Y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            # Show image from the camera, each frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("General exception: ", e)

    # Shows web cam window
    webcam.release()
    cv2.destroyAllWindows()


def main():
    show_Cam()


if __name__ == "__main__":
    main()
