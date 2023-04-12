# Child Height Prediction Model
This code provides a solution to predict the height of children based on depthmap images and corresponding pose key points. It uses a deep learning approach to train a model on the given dataset and then use it for height prediction.

## Prerequisites
- Python 3
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
- Pandas
- Seaborn
- Matplotlib


## Dataset
The dataset consists of depthmap images and corresponding pose key points and height of the children. The images are preprocessed and loaded into train_inputs and val_inputs variables. The corresponding heights are loaded into train_outputs and val_outputs variables.

## Getting Started
1. Clone this repository to your local machine.
2. Download the height_and_pose.xlsx file and save it in the root directory of the repository.
3. Download the depthmap folder containing the depthmap images and save it in the root directory of the repository.
4. Install the required Python packages using pip or conda.
```bash
pip install opencv-python numpy pandas tensorflow numpy pandas seaborn matplotlib
```
5. Open the provided Jupyter notebook - 'depthmap_model.ipynb' on Google Colab or locally and run the code cells sequentially.
6. Make sure to change the paths to the dataset and any other relevant paths as per your local setup.
7. Once the model has been trained, you can use it to predict the height of children given a new depthmap and pose key points.

## Approach
The solution uses a Convolutional Neural Network (CNN) to learn the features from the depthmap images and pose key points. The output of the CNN is then passed through a few fully connected layers to predict the height of the child.

![nn](https://user-images.githubusercontent.com/79955028/231344502-11885cad-8a4e-4390-801c-f5b8bad420a2.svg)

## Model Architecture
The model architecture consists of the following layers:

- Three convolutional layers with 32, 64, and 128 filters and kernel size of (3,3) and ReLU activation function. Max pooling is applied after each convolutional layer.
- Flatten layer to convert the output of the convolutional layers into a 1D array.
- Two fully connected layers with 128 and 64 neurons and ReLU activation function.
- Output layer with a single neuron and linear activation function.

## Reasoning for Deep Learning based CNN selection
1. The selected deep learning algorithm for predicting estimated heights in a person is a Convolutional Neural Network (CNN). CNNs are often used in image recognition tasks, as they are specifically designed to handle the spatial relationships present in images.

2. In this case, the input data consists of depth maps, which are essentially 2D images. The CNN architecture used in the code snippet includes multiple convolutional layers, which are able to identify various features in the input image, followed by max pooling layers, which reduce the spatial dimensions of the output.

3. After several convolutional and pooling layers, the output is flattened and passed through fully connected layers, which can learn to combine the various features extracted from the input. Finally, a single output neuron is used with a linear activation function to predict the estimated height of the person.

4. Overall, the CNN architecture is a good choice for this task because it is designed to handle spatial data like images, and is capable of learning complex features that are important for estimating the height of a person.

## Training
The model is trained for 30 epochs using the Adam optimizer and Mean Squared Error (MSE) loss function. The model is compiled with Root Mean Squared Error (RMSE), MSE, and Mean Absolute Error (MAE) metrics to monitor the training progress. The training was performed using Google Colab.

## Prediction
After training the model, it is saved to a file named model.h5. To predict the height of a child, first, a depthmap image and corresponding pose key points are fed to the trained model, and the output is the predicted height of the child.

## Conclusion
In conclusion, this solution provides a deep learning approach to predict the height of children based on depthmap images and corresponding pose key points. The model is trained on the given dataset using a CNN architecture and is evaluated using various metrics to monitor the training progress. The trained model can be used to predict the height of a child given a depthmap image and corresponding pose key points.



