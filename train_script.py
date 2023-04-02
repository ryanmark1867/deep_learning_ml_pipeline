import os
import time
import seaborn as sns
# import datetime, timedelta
import datetime
import pydotplus
from datetime import datetime, timedelta
from datetime import date
from dateutil import relativedelta
from io import StringIO
import pandas as pd
import pickle
from pickle import dump
from pickle import load
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
# DSX code to import uploaded documents
from io import StringIO
import requests
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import yaml
import math


project_id = 'first-project-ml-tabular'
pipeline_root_path = '/home/ryanmark2023/ml_pipeline'

# get values from environment variables defined by Vertex AI in the training container
# using details from https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline
# and https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets
# and https://towardsdatascience.com/developing-and-deploying-a-machine-learning-model-on-vertex-ai-using-python-865b535814f8
# and https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/edition2/10_mlops/model.py



#from google.cloud import aiplatform
#from google_cloud_pipeline_components import aiplatform as gcc_aip

project_id = 'first-project-ml-tabular'
pipeline_root_path = '/home/ryanmark2023/ml_pipeline'

OUTPUT_MODEL_DIR = os.getenv("AIP_MODEL_DIR") 
TRAIN_DATA_PATTERN = os.getenv("AIP_TRAINING_DATA_URI")
EVAL_DATA_PATTERN = os.getenv("AIP_VALIDATION_DATA_URI")
TEST_DATA_PATTERN = os.getenv("AIP_TEST_DATA_URI")

print("TRAIN_DATA_PATTERN: ",TRAIN_DATA_PATTERN)
print("EVAL_DATA_PATTERN: ",EVAL_DATA_PATTERN)
print("TEST_DATA_PATTERN: ",TEST_DATA_PATTERN)
