#import kfp
import os
import logging
import argparse
from google.cloud import storage
import yaml
#from google.cloud import aiplatform
#from google_cloud_pipeline_components import aiplatform as gcc_aip
logging.getLogger().setLevel(logging.INFO)
project_id = 'first-project-ml-tabular'
pipeline_root_path = '/home/ryanmark2023/ml_pipeline'
parser = argparse.ArgumentParser()
parser.add_argument(
        '--config_bucket',
        help='Config details',
        required=True
    )
args = parser.parse_args().__dict__
config_bucket = args['config_bucket']
# use the method described here to get parts of URI https://engineeringfordatascience.com/posts/how_to_extract_bucket_and_filename_info_from_gcs_uri/
bucket_name = config_bucket.split("/")[2]
object_name = "/".join(config_bucket.split("/")[3:])
# read the object https://cloud.google.com/appengine/docs/legacy/standard/python/googlecloudstorageclient/read-write-to-cloud-storage
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(object_name)
with blob.open("r") as f:
    config = yaml.safe_load(f)
print("config is: ",config)

'''
OUTPUT_MODEL_DIR = os.getenv("AIP_MODEL_DIR") 
TRAIN_DATA_PATTERN = os.getenv("AIP_TRAINING_DATA_URI")
EVAL_DATA_PATTERN = os.getenv("AIP_VALIDATION_DATA_URI")
TEST_DATA_PATTERN = os.getenv("AIP_TEST_DATA_URI")

logging.info("args dict: ",args)
logging.info("about to put out values")
logging.info("OUTPUT_MODEL_DIR: ",OUTPUT_MODEL_DIR)
logging.info("TRAIN_DATA_PATTERN: ",TRAIN_DATA_PATTERN)
logging.info("EVAL_DATA_PATTERN: ",EVAL_DATA_PATTERN)
logging.info("TEST_DATA_PATTERN: ",TEST_DATA_PATTERN)
logging.info("Done")
'''
