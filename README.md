# Flower Classification Model with CNN
This is my neural network model for [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset). 
I decide to code this neural network for my personal learning.

## Table of Contents

- [Installation](#installation)
- [Running the Project](#running-the-project)
- [About Dataset](#about-dataset)
- [About the Convolutional Neural Network (CNN)](#About-the-Convolutional-Neural-Network-(CNN))
- [Results](#results)
- [Diagrams](#diagrams)


# Installation
Clone the repository to your local machine and install requirements.txt:
```
git clone https://github.com/dim-tsoukalas/flower_classification_ai.git
cd flower_classification_ai/code
pip install -r requirements.txt
```
# Running the Project
Go to /code folder and run the main.py file:
```
cd flower_classification_ai/code
python main.py
```
# About Dataset
This dataset belongs to [DPhi Data Sprint #25: Flower Recognition](https://dphi.tech/practice/challenge/61). The dataset contains raw jpeg images of five types of flowers.

- daisy
- dandelion
- rose
- sunflower
- tulip
  
<img width="809" alt="Screenshot 2024-09-20 at 10 33 24 PM" src="https://github.com/user-attachments/assets/6966fa10-e923-4f81-adee-ee024ce1b737">

# About the Convolutional Neural Network (CNN) 
This CNN can process images with size 112x112x3 and can classifie them to 5 classes.
In my CNN I used:

<img width="643" alt="Screenshot 2024-09-20 at 10 44 15 PM" src="https://github.com/user-attachments/assets/24194486-70df-458b-870c-ee2c29ba834d">


# Results
Training Set Acc | Test Set Acc
--- | --- 
0.8 | 0.78

# Diagrams
Below are the diagrams of the training and evaluation **Accuracy-Epoch**, **Loss-Epoch**

![__results___7_0](https://github.com/user-attachments/assets/e1800e9c-0543-4a0b-be90-20844ab0b499)


