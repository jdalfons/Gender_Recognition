# Model trainer

* **category**    Machine learning  📡
* **author**      Juan Diego Alfonso <jalfons.ocampo@gmail.com>
* **copyright**   [GNU General Public Licence](https://www.gnu.org/licenses/gpl.txt)
* **source**  [GitHub](https://github.com/halcolo/Model_trainer.git)


## Description

This model was created to explain how to recognice the gender of a person based in the face using CNN(Convolutional Neural network) with Keras.

## Virtual enviroment

You can create a virtual environment using following commands.

In linux maybe you need the following package `sudo apt-get install python3-venv`

| Description|Linux| Win|
| ------ | ------ |------|
| Install virtenv| `python3 -m pip install --user virtualenv` |`py -m pip install --user virtualenv`|
| Create enviroment | `python3 -m venv modelTrainerEnv` |`py -m venv modelTrainerEnv`|
| Activate enviroment | `source modelTrainerEnv/bin/activate` |`.\modelTrainerEnv\Scripts\activate`|
| Install packages | `pip install --upgrade -r requirements.txt`  |`pip install --upgrade -r requirements.txt` |
| Inactivate  | `deactivate`  | `deactivate`  |

## USE

This project is a CNN based in a large-scale image recognition training model, as called VGG, this model is based in layers and a compression of an image, the schema is explained in the following graph:

 ![:blurdata:](https://github.com/halcolo/gender_recognition/blob/master/img/VGG_genderRecog.png?raw=true ":blurdata:")

### Training

To train the model just run with python 3.6 and the environment created the file __training.py__, ensure the route of the 'datadir' variable is set as your dataset photos is.
To download the dataset you can see follow URL.
https://storage.googleapis.com/pretrained-model-gender/gender_dataset_face.rar

### Recognition

To run the recognition ensure the model is in the folder 'model' and run __detection.py__ and comment the variable 'model_path', if you haven't the model the app download a model automatically.


| Aplication| Version|
| ------ | ------ |
| Python| `Python 3.6` |
| OpenCV| `4.1.0` |
| numpy | `1.16.3` |
| tensorflow | `1.13`  |
| keras   | `2.2.4` |
| cvlib   | `0.2.1` |
| scikit-learn   | `0.21.1` |
| matplotlib| `3.0.3` |
