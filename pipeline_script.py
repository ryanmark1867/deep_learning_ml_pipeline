# script to drive training process for a Keras model trained on tabular data in Vertex AI
# adapts ideas from https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/edition2/10_mlops/train_on_vertexai.py

# imports
from google.cloud import aiplatform
import tensorflow as tf
import argparse
import yaml
import os
from datetime import datetime


TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")




def get_pipeline_config(path_to_yaml):
    print("path_to_yaml "+path_to_yaml)
    try:
        with open (path_to_yaml, 'r') as c_file:
            config = yaml.safe_load(c_file)
    except Exception as e:
        print('Error reading the config file')
    return config
 

# define CustomTrainingJob object
def create_job(config):
    model_display_name = '{}-{}'.format(config['ENDPOINT_NAME'], TIMESTAMP)
    job = aiplatform.CustomTrainingJob(
            display_name='train-{}'.format(model_display_name),
            script_path = config['script_path'],
            container_uri=config['train_image'],
            staging_bucket = config['staging_path'],
            requirements=['gcsfs'],  # any extra Python packages
            model_serving_container_image_uri=config['deploy_image']
    ) 
    # define dataset
    return job


# run job to create dataset
def run_job(job, ds, model_args,config):
    model_display_name = '{}-{}'.format(config['ENDPOINT_NAME'], TIMESTAMP)
    model = job.run(
        dataset=ds,
        # See https://googleapis.dev/python/aiplatform/latest/aiplatform.html#
        training_fraction_split = config['training_fraction_split'],
        validation_fraction_split = config['validation_fraction_split'],
        test_fraction_split = config['test_fraction_split'],
        model_display_name=model_display_name,
        args=model_args,
    #    replica_count=1,
        machine_type= config['machine_type']
        # See https://cloud.google.com/vertex-ai/docs/general/locations#accelerators
    #    accelerator_type=aip.AcceleratorType.NVIDIA_TESLA_T4.name,
    #    accelerator_count=1,
    #    sync=develop_mode
    )
    return model


if __name__ == '__main__':
    # load pipeline config parameters
    config = get_pipeline_config('pipeline_config.yml')
    # all the arguments sent to the training script run in the container are sent via
    # a yaml file in Cloud Storage whose URI is the single argument sent
    model_args = ['--config_bucket', config['config_bucket_path']]
    print("model_args: ",model_args)
    job = create_job(config)
    dataset_path = 'projects/'+config['project_id']+'/locations/'+config['region']+'/datasets/'+config['dataset_id']
    ds = aiplatform.TabularDataset(dataset_path)
    model = run_job(job, ds, model_args,config)


