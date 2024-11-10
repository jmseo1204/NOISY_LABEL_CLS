# Image classification with label noise on CIFAR100 dataset

## Introduction

We will implement neural network models for image classification problem with label noise and class imbalance.

The second assignment let you implement any neural network models for image classification on CIFAR100 dataset with noisy label.
Noisy label here means that some of the given labels are not correct (e.g., it is a dog image but labeled as a cat).

We recommend you to use PyTorch as your deep learning libary but we welcome you to use other deep learning libraries (e.g., Tensorflow).

**Note**: we will use `Python 3.x` for the project. (We have tested all codes with Python 3.7.7)

---
## Preparation

### Installing prerequisites

The prerequisite usually refers to the necessary library that your code can run with. They are also known as `dependency`. We have prepared a few libraries for you to start with. To install the prerequisite, simply type in the shell prompt (not in a python interpreter) the following:

```
$ pip install -r requirements.txt
```

### Download the dataset

Go to [dataset page in our kaggle page for this challenge (CIFAR100-NoisyLabel)](https://www.kaggle.com/c/cifar100-image-classification-with-noisy-labels/data) to download the dataset. Copy (or move) the dataset into `./dataset` sub-directory.


---
## Prepare the dataset

Read the csv to load the dataset.

```
>>> import datasets
>>> dataset_nl = datasets.C100Dataset('./dataset/data/cifar100_nl.csv')
>>> [data_nl_tr_x, data_nl_tr_y, data_nl_val_x, data_nl_val_y] = dataset_odn.getDataset()
```

Each line of the dataset's `csv` file follows the below format:
```
filename,classname
```

*Note*: You may ignore the warning of `Fontconfig warning: ignoring UTF-8: not a valid region tag` after `import datasets` command.

**Note**: Do not use PyTorch's CIFAR100 dataset loader. Use your own dataset loader.

---
## Image classification with dataset of noisy labels

Perform image classification using the given dataset of noisy label version of CIFAR-100 dataset (`cifar100_nl.csv`).

`REPORT1`: Describe model your have used (1. architecture overview, 2. any specialty of this model and etc.)

`REPORT2`: Report both the training and testing accuracy in a plot (x: epoch, y: accuracy). 

`REPORT3`: Discuss any ideas to improve the accuracy (e.g., new architecture, using new layers, using new loss)

