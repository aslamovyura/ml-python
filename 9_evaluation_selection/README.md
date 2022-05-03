# Capstone Project

Capstone project for RS School Machine Learning course.

This demo uses [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/) dataset.

## Usage
This package allows you to train model for forest categories classification.
1. Clone this repository to your machine.
2. Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine.
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. To generate dataset profiling report (EDA), use the following command:
```sh
poetry run dataset_profiling -d <path to csv with data>
```
or if you just want to visualize dataset:
```sh
poetry run dataset_gui -d <path to csv with data>
```

6. Run training process with the following commands:
- Logistic Regression:
```sh
poetry run train_logistic -d <path to csv with data> -s <path to save trained model>
```
- Random Forest Classifier:
```sh
poetry run train_rfc -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train_logistic --help
```
7. Run training process with optimal hyperparameters search with the following commands:
- Logistic Regression:
```sh
poetry run train_logistic_opt -d <path to csv with data> -s <path to save trained model>
```
- Random Forest Classifier:
```sh
poetry run train_rfc_opt -d <path to csv with data> -s <path to save trained model>
```
8. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
Experiment results for Logistic Regression and Random Forest Classifier are stored in different folders:
![logreg_exp](assets/logreg_experiments.png)
![rfc_exp](assets/rfc_experiments.png)

## Development

The code in this repository is tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox [-r]
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
nox -[r]s black
poetry run black src tests noxfile.py
```
