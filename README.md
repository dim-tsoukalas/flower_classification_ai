# Flower Classification Model with CNN
This is my neural network model for [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset). 
I decide to code this neural network for my personal learning.

## About Dataset
This dataset belongs to [DPhi Data Sprint #25: Flower Recognition](https://dphi.tech/practice/challenge/61). The dataset contains raw jpeg images of five types of flowers.

- daisy
- dandelion
- rose
- sunflower
- tulip

### Content
- train - contains all the images that are to be used for training your model.  In this folder you will find five folders namely - 'daisy', 'dandelion', 'rose', 'sunflower' and 'tulip' which contain the images of the respective flowers
- test - contains 924 flowers images. For these images you are required to make predictions as the respective flower names - 'daisy', 'dandelion', 'rose', 'sunflower' and 'tulip'
- Testing_set_flower.csv - this is the order of the predictions for each image that is to be submitted on the platform. Make sure the predictions you download are with their image's filename in the same order as given in this file.
sample_submission: This is a csv file that contains the sample submission for the data sprint.

## About the Convolutional Neural Network (CNN)
This CNN can process images with size 112x112x3 and can classifie them to 5 classes.
In my CNN I used:

Neuron | Size
--- | --- 
Input | (112, 112, 3)
#1 | 32, (3,3)
#2 | 64, (3,3)
#3 | 128, (3,3)
#4 | 512
Output | 5

### Results
Training Set Acc | Test Set Acc
--- | --- 
0.8 | 0.78

### Diagrams
Below are the diagrams of the training and evaluation **Accuracy-Epoch**, **Loss-Epoch**

![__results___7_0](https://github.com/user-attachments/assets/e1800e9c-0543-4a0b-be90-20844ab0b499)


