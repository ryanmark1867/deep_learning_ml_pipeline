#import kfp
import os
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
