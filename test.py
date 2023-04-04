#import kfp
import os
import logging
import argparse
from google.cloud import storage
import yaml
import glob
import pandas as pd
import fnmatch
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
# arg to use on command line: "gs://third-project-ml-tabular-bucket/training_scripts/model_training_config.yml"
# use the method described here to get parts of URI https://engineeringfordatascience.com/posts/how_to_extract_bucket_and_filename_info_from_gcs_uri/
bucket_name = config_bucket.split("/")[2]
object_name = "/".join(config_bucket.split("/")[3:])
# read the object https://cloud.google.com/appengine/docs/legacy/standard/python/googlecloudstorageclient/read-write-to-cloud-storage
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob_out = bucket.blob(object_name)
print("config_bucket",config_bucket)
print("bucket_name",bucket_name)
print("object_name",object_name)

with blob_out.open("r") as f:
    config = yaml.safe_load(f)
print("config is: ",config)
stringer = blob_out.download_as_string()
stringer_string = stringer.decode("utf-8")
print("stringer_string is: ", stringer_string)

# gs://second-project-ml-tabular-bucket/staging/aiplatform-custom-training-2023-04-04-14:53:53.021/dataset-2901415472631119872-tables-2023-04-04T14:53:53.487135Z/test-00000-of-00004.csv
#tracer_pattern = "gs://second-project-ml-tabular-bucket/staging/aiplatform-custom-training-2023-04-04-14:53:53.021/dataset-2901415472631119872-tables-2023-04-04T14:53:53.487135Z/training-00001-of-00004.csv"
tracer_pattern = "gs://second-project-ml-tabular-bucket/staging/aiplatform-custom-training-2023-04-04-14:53:53.021/dataset-2901415472631119872-tables-2023-04-04T14:53:53.487135Z/test-*.csv"
bucket_pattern = tracer_pattern.split("/")[2]
pattern = "/".join(tracer_pattern.split("/")[3:])
print("pattern is: ",pattern)
#csv_files = glob.glob("gs://second-project-ml-tabular-bucket/staging/aiplatform-custom-training-2023-04-04-14:53:53.021/dataset-2901415472631119872-tables-2023-04-04T14:53:53.487135Z/test-*")
#csv_files = glob.glob(tracer_pattern)
pattern_client = storage.Client()
bucket = pattern_client.get_bucket(bucket_pattern)
blobs = bucket.list_blobs()
matching_files = [f"gs://{bucket_pattern}/{blob.name}" for blob in blobs if fnmatch.fnmatch(blob.name, pattern)]
print("matching_files is: ",matching_files)
df = pd.concat([pd.read_csv(f) for f in matching_files], ignore_index=True)
print("df shape is",df.shape)

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
