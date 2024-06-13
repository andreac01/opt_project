# Repository README

This repository contains the code for the Oprimization For Machine Learning Course Mini-Project.

## Description

**Abstract:** In this project, a methodology to obtain a visual representation of loss functions is presented. 
These representations allow for a comparison between the behaviors of various optimization algorithms in real-world scenarios rather than just simple problems where the function to approximate is fully known. After defining this methodology, classical gradient descent (GD), stochastic gradient descent (SGD), and Adaptive Moment Estimation (Adam) are compared in different learning rate and batch size scenarios to highlight their differences.
A more complete description of the project is present in `report.pdf` file.

## Installation

To install all dependencies run `pip install -r requirements.txt`

## Code

- **plottings.py** containing all the function for calculating trajectories, loss and plot them
- **networks.py** contain the network definition and utility function for training
- **gd.py, sgd.py, adam.py** used to obtain results about the optimizer comparison and hyperparameter analysis. Each of these codes train the network with a different optimizer and save relative images. The hyperparameters need to be setted into each file. 
As an example run `python adam.py" to train the network using ada optimizer.
- **create_loss_approx.py** used to generate the plain loss function plots used in the first presented section of the results in report.pdf. 
It can be simply run as `python create_loss_approx.py`, hyperparameters can be changed in the script.


## Simplified repository structure

opt_project:
- adam.py
- create_loss_approx.py
- data
  - CIFAR10 dataset
- gd.py
- images 
  - Images used in report
- load_interactive.py
- network.py
- outputs 
  - outputs from training
- plots
  - plain2D
    - Plots of the loss function for different batch sizes
  - plain3D
    - 3D Plots of the loss function for different batch sizes
  -SimpleNN
      - Plots for GD, SGD and Adam trainings

- plottings.py
- README.md
- report.pdf
- requirements.txt
- sgd.py



