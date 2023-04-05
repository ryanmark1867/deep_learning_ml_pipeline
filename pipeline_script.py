# script to drive training process for a Keras model trained on tabular data in Vertex AI

# imports
from google.cloud import aiplatform
import tensorflow as tf
import argparse
import yaml
import os
from datetime import datetime


current_path = os.getcwd()
print("current directory is: "+current_path)
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
ENDPOINT_NAME = 'klrealestate'
# adapted from https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/edition2/10_mlops/train_on_vertexai.py
# this fails because it assumes a single digit second level for tf level, so you get back to 2.1 instead of 2.11
tf_version = '2-' + tf.__version__[2:3]
# for simplicity's sake, hardcode images that match the images used training in Colab
train_image = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-9:latest"
deploy_image = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest"
print("train_image is: ",train_image)
print("deploy_image is: ",deploy_image)
print("tf.__version__ is:",str(tf.__version__))
# assumes that config file is in same directory as this script
config_file_path = "model_training_config.yml"
config_bucket_path = "gs://third-project-ml-tabular-bucket/training_scripts/model_training_config.yml"
script_path = "model_training_keras_preprocessing.py"
machine_type = 'n1-standard-4'


project_id = 'first-project-ml-tabular'
region = 'us-central1'
dataset_id = '2901415472631119872'
dataset_path = 'projects/'+project_id+'/locations/'+region+'/datasets/'+dataset_id
staging_path = "gs://second-project-ml-tabular-bucket/staging/"
 
print("dataset_path is: ",dataset_path)
# define CustomTrainingJob object
def create_job():
    model_display_name = '{}-{}'.format(ENDPOINT_NAME, TIMESTAMP)
    job = aiplatform.CustomTrainingJob(
            display_name='train-{}'.format(model_display_name),
            script_path = script_path,
            container_uri=train_image,
            staging_bucket = staging_path,
            requirements=['gcsfs'],  # any extra Python packages
            model_serving_container_image_uri=deploy_image
    ) 
    # define dataset
    return job


# run job to create dataset
def run_job(ds, model_args):
    model_display_name = '{}-{}'.format(ENDPOINT_NAME, TIMESTAMP)
    model = job.run(
        dataset=ds,
        # See https://googleapis.dev/python/aiplatform/latest/aiplatform.html#
        training_fraction_split = 0.8,
        validation_fraction_split = 0.1,
        test_fraction_split=0.1,
        model_display_name=model_display_name,
        args=model_args,
    #    replica_count=1,
        machine_type= machine_type
        # See https://cloud.google.com/vertex-ai/docs/general/locations#accelerators
    #    accelerator_type=aip.AcceleratorType.NVIDIA_TESLA_T4.name,
    #    accelerator_count=1,
    #    sync=develop_mode
    )
    return model


# load config file into Python dictionary
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

# convert yaml dictionary to argparse object
def create_argparser_from_yaml(yaml_data):
    parser = argparse.ArgumentParser()
    
    for key, value in yaml_data.items():
        arg_type = type(value)
        if arg_type == bool:
            action = 'store_false' if value else 'store_true'
            parser.add_argument(f'--{key}', dest=key, action=action, default=value)
        else:
            parser.add_argument(f'--{key}', dest=key, type=arg_type, default=value)
    
    return parser


if __name__ == '__main__':
    config_dict = load_yaml(config_file_path)
    parser = create_argparser_from_yaml(config_dict)
    # list(inputDictionary.items())
    # all the arguments sent to the training script run in the container are sent via
    # a yaml file in Cloud Storage whose URI is the single argument sent
    model_args = ['--config_bucket', config_bucket_path]
    print("model_args: ",model_args)
    job = create_job()
    ds = aiplatform.TabularDataset(dataset_path)
    print("model_args: ",model_args)
    model = run_job(ds, model_args)


