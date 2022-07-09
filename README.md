# Imagine Texting & Driving

Imagine Texting & Driving is a convolutional neural network that can be used to automate the task of deducing whether 
a person is texting while driving or not.

## The Algorithm

The model is a convolutional neural network powered by Keras, meaning it is specifically designed to classify images. In this case, images 
of people driving. It is trained using a modified version of [this dataset](https://www.kaggle.com/datasets/sarahh222/distracteddriversrevampeddataset) 
(which you can download by clicking [here](https://drive.google.com/file/d/1LJQSB9yx0qGXZS0NUrPvpJCI2wh6FPtC/view?usp=sharing).

The images are reduced to a resolution of 256x256 and prefetched to optimize loading the dataset. The dataset has 5
classes: normal, which is just normal driving, and 4 variations of texting or calling while driving. Images from the dataset
are fed into the training process with all 3 color channels retained. The model structure itself has several layers, 
including a dropout layer to prevent overfitting.

Softmax is the preferred activation function for the final layer of the model, this is the most common activation function for 
multi-class classification models, as opposed to the sigmoid function used for binary classification. Various callbacks 
are put in place to monitor the training process, including early stopping and model checkpointing. Categorical 
crossentropy is used for the loss function since it's best suited for multi-class classification.

## Running this project

### Training the model
1. [Follow this guide to install Tensorflow on the Jetson Nano](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
2. [Download the dataset here](https://drive.google.com/file/d/1LJQSB9yx0qGXZS0NUrPvpJCI2wh6FPtC/view?usp=sharing) and place it in the root directory of this project.
3. Run `python3 train.py` and wait for training to complete.

### Picking the best model
The script `train.py` will create a new model and store it in `checkpoints/` after every epoch and 
automatically stop training once overfitting is detected, however you still need to
pick from the latest few iterations of the model to minimize undetected overfitting.
1. Observe the output of the `train.py` script.
   ![train.py output](https://i.imgur.com/Eba1cI8.png)
2. Notice how the validation loss keeps decreasing as epochs progress. Usually, a sudden sharp increase in the 
validation loss indicates that the model is overfitting. Choose the model in `checkpoints/` that corresponds to the
epoch with the smallest validation loss.

### Testing
Run `python3 test.py` to test the model on the test set. Edit the `class_to_test` variable to test the accuracy of a 
different class in the test dataset. Remember to also edit the `model_path` variable to match the model you want to test.

### Proof of Concept
An interesting application of this model is to detect whether a taxi driver is texting while driving or not to 
ensure the safety of passengers. Taxi services could attach a camera facing the driver which periodically takes
a photo and sends it to a server where the model can detect whether the driver is texting or not. This simple
web server made with FastAPI is a demo if this concept.
1. Run `pip install fastapi uvicorn`.
2. Edit the `model_path` variable to point to the model you want to use, just like you did with `test.py`.
3. Run `uvicorn server:app`.
4. You will see the line `Uvicorn running on <url> (Press CTRL+C to quit)` in the terminal. Visit the url in your browser.

[View a video explanation here](video link)