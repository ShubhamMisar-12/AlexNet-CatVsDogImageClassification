# Cat vs Dog

[Link to Kaggle](https://www.kaggle.com/code/shubhammisar/cat-vs-dog-alex-net-pytorch)

Cat vs. Dog Image Classification using AlexNet in PyTorch

![Image](https://images.unsplash.com/photo-1450778869180-41d0601e046e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1286&q=80)

## Overview

This project implements a machine learning model for classifying images as either cats or dogs using the AlexNet architecture in PyTorch. By training on a large dataset of labeled cat and dog images, the model learns to differentiate between the two classes and make predictions on unseen images.

## Folder Structure

- Files
  - Data
    - local data
  - LICENSE
  - README.md
  - CatVsDog.ipynb
  - requriemets.txt

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository.
2. Navigate to the project directory.
3. Install the required dependencies by running the following command:

```bash
pip install -r requriemets.txt
```

4. Open the `CatVsDog.ipynb` notebook to view and run the code.

## Project Details

# Cats and Dogs Image Classification

This repository contains code for training an image classification model to distinguish between cats and dogs using the Cats and Dogs dataset.

## Data Preparation

The dataset consists of images of cats and dogs. The data is organized into training and test sets, with separate directories for cats and dogs in each set. The dataset can be found in the `/kaggle/input/cats-and-dogs-image-classification` directory.

To prepare the data, the following steps are performed:

1. Load the dataset using the `ImageFolder` class from the `torchvision.datasets` module.
2. Apply transformations to the images, including resizing, random flips, and converting to tensors.
3. Calculate the mean and standard deviation of the training dataset for normalization.

## Model Architecture

The model architecture used for image classification is `AlexNet`, which is a convolutional neural network. The model consists of several convolutional and pooling layers, followed by fully connected layers. The model is implemented using the `nn.Module` class from the `torch.nn` module.

## Training

The model is trained using the following steps:

1. Set up hyperparameters, including the learning rate, number of epochs, and optimizer.
2. Create a data loader for the training and test datasets.
3. Perform the training loop, iterating over the epochs.
   - In each epoch, iterate over the batches of training data.
   - Perform the forward pass, compute the loss, and update the model parameters.
   - Track the training loss and accuracy.
4. Evaluate the model on the test dataset after each epoch.
   - Compute the test loss and accuracy.
   - Track the test accuracy.

## Results

The training process produces the following results:

- Train Loss: 0.5689 - Train Acc: 0.7469 - Test Loss: 0.6512 - Test Acc: 0.6571

The model achieves a test accuracy of approximately 65.71% after training for 100 epochs.

## Requirements

The code is implemented in Python 3 and requires the following libraries:

- NumPy
- Pandas
- PyTorch
- torchvision
- Matplotlib
- tqdm

You can install the required libraries by running `pip install -r requirements.txt`.

## License

This project is licensed under the MIT License.

## Usage

To train the model, run the `train.py` script:
