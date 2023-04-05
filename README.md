# ML pipeline for a Trained Deep Learning Model

This repo contains the code for a simple Kubeflow ML pipeline for a trained deep learning model for the upcoming Manning book on machine learning with tabular data. This is part of the code for the end-to-end deep learning and MLOps chapter. This code is designed to be run from Cloud Shell in Google Cloud.

File descriptions 

- [model_training_keras_preprocessing.py](https://github.com/ryanmark1867/deep_learning_ml_pipeline/blob/master/model_training_keras_preprocessing.py): model training script - adapted from the notebook version of the training code: [model_training_keras_preprocessing.ipynb](https://github.com/ryanmark1867/deep_learning_best_practices/blob/master/notebooks/model_training_keras_preprocessing.ipynb)
- [model_training_config.yml](https://github.com/ryanmark1867/deep_learning_ml_pipeline/blob/master/model_training_config.yml): config file for model_training_keras_preprocessing.py. This file is not accessed directly. Instead, it is copied to Google Cloud Storage and the URI for that blob is passed as an argument to the training code running in a container.
- [pipeline_script.py](https://github.com/ryanmark1867/deep_learning_ml_pipeline/blob/master/pipeline_script.py): script for Kubeflow pipeline that invokes model_training_keras_preprocessing.py in a pre-built Vertex AI container
- [pipeline_config.yml](https://github.com/ryanmark1867/deep_learning_ml_pipeline/blob/master/pipeline_config.yml): config file containing parameters for pipeline_script.py

Here are the articles that describe the deployments in more detail:

- TBD: here

