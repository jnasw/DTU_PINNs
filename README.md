# Physics-Informed-Neural-Networks-for-Synchronous-Machine-Models

This repository contains the project work for the thesis: Physics-Informed-Neural-Networks-for-Synchronous-Machine-Models.


The following provides a brief overview on how to install the required packages and how to get started.

## Installation

To setup the python environment, an environemnt `environment.yml` is needed
```
conda env create -f environment.yml
conda activate pinns
```

Next, install all the necessary packages by navigating to the folder containing this repository and executing the following command:

```
pip install -r requirements.txt
```
WANDB api_key  
In src/conf, replace the value of api_key in the various setup_dataset and setup_dataset_nn YAML files with your own key obtained from  [Weights & Biases (wandb)](https://wandb.ai/).

## Getting started

To illustrate the main functionalities of the two developed pipelines, we provide a few examples in `jupyter_notebooks`. They contain references to central files and should help to explore the repository.

The repository has two major parts
- `The pipeline for the solution of the Ordinary Differential Equations (ODEs)` provides the respective solutions and equations for the selected machine number and synchronous machine modelling
- `The Physics-Informed Neural Network (PINN) pipeline` provide the functionality that is needed to create, train and test the neural network for the given solutions/equations from the previous pipeline.


## Using the frameworks 

The straightforward implementation, including dataset/solution generation and the training of neural networks, is available. All configuration files for this implementation are located in the src/conf folder.

To create the dataset, execute ``create_dataset_d.py``. This script uses parameters located in ``src/conf/setup_dataset.yaml``, which specifies all the configurations. Comments in the YAML file display all available options. Different initial conditions for each modeling can be adjusted in the YAML files located in the src/conf/initial_conditions folder.

To train the Physics-Informed Neural Networks (PINNs), ``execute test_sweep.py``. Utilizing Weights & Biases (wandb) sweep, all combinations of hyperparameters are executed. Similar to the previous step, the configuration file ``src/conf/setup_dataset_nn.yaml`` contains all the different parameters and hyperparameters. It is important that these two main configuration files are aligned regarding the modeling and the machine number.

For immediate implementation of the above steps in a High-Performance Computing (HPC) environment, use the provided script jobscript.sh.

## Output/Results

The dataset with the solutions and the weights of the Neural Networks are saved in the desired folder, specified in the configuration files.
The results from the training are logged and visualized within [Weights & Biases (wandb)](https://wandb.ai/) project: Project_PINN_for_synchronous_machine.

