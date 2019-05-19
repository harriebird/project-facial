# project-facial
A simple facial recognition system using Histogram of Gradients and Artificial Neural Network.

The project-facial has three parts: **registration**, **training**  and **recognition**.
* The **registration** part is used to collect images and its labels, saving it to the `images` directory.
* The **training** part is used to iterate on `images` directory and retrieves all the captured facial images including its corresponding label. Those retrieved images will be searched for faces and compute its HOG values. Those detected faces with its corresponding label will be fed into MLP Classifier for the actual training. After the training, the trained model and the labels will be saved into the `training_result` directory.
* The **recognition** part is used to recognize faces captured by the camera based on the trained model produced by the by the training script.

## Requirements
1. **Python 3**
2. **OpenCV**
3. **scikit-learn**

**Python 3 must be installed on the machine** where this will be used. You can get it [here](https://www.python.org/downloads/).

Use of **virtual environment** is highly recommended. If want to learn how to use it, you can visit the [Python documentation](https://docs.python.org/3/tutorial/venv.html).

## Installation
1. Clone or download the repository.
2. Change the current directory to where the project root directory is.
3. Create a Python 3 virtual environment using the command `python3 -m venv myvenv` and activate it.
4. Install the required libraries by using the command `pip install -r requirements.txt`.

## Using the project-facial
* To capture and register faces using camera, run `python register.py`
* To start the training, run `python train.py`
* To start the facial recognition, run `python recognition.py`
