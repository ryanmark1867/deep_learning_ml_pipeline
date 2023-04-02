from google.cloud import aiplatform
print("just before tf import")
import tensorflow as tf
print("just after tf import")

# adapted from https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/edition2/10_mlops/train_on_vertexai.py
tf_version = '2-' + tf.__version__[2:3]
train_image = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.{}:latest".format(tf_version)
# us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-9:latest
deploy_image = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.{}:latest".format(tf_version)
model_display_name = "kuala Lumpur real estate prediction"

# RuntimeError: staging_bucket should be set in TrainingJob constructor 
# or set using aiplatform.init(staging_bucket='gs://my-bucket')
# define dataset object using parms from dataset defined in Vertex AI UI
project_id = 'first-project-ml-tabular'
region = 'us-central1'
dataset_id = '2901415472631119872'
dataset_path = 'projects/'+project_id+'/locations/'+region+'/datasets/'+dataset_id
print("dataset_path is: ",dataset_path)
# define CustomTrainingJob object
job = aiplatform.CustomTrainingJob(
        display_name='train-{}'.format(model_display_name),
#        script_path="gs://first-project-ml-tabular-bucket/training_scripts/test.py",
        script_path = "test.py",
#        container_uri=train_image,
        container_uri=train_image,
        staging_bucket = "gs://second-project-ml-tabular-bucket/staging/",
        requirements=[],  # any extra Python packages
        model_serving_container_image_uri=deploy_image
) 
# define dataset
ds = aiplatform.TabularDataset(dataset_path)
# run job to create dataset
model = job.run(
    dataset=ds,
    # See https://googleapis.dev/python/aiplatform/latest/aiplatform.html#
    training_fraction_split = 0.8,
    validation_fraction_split = 0.1,
    test_fraction_split=0.1,
#    predefined_split_column_name='data_split',
    model_display_name="test pipeline",
#    args=model_args,
#    replica_count=1,
    machine_type='n1-standard-4'
    # See https://cloud.google.com/vertex-ai/docs/general/locations#accelerators
#    accelerator_type=aip.AcceleratorType.NVIDIA_TESLA_T4.name,
#    accelerator_count=1,
#    sync=develop_mode
)
#job.run( ds, replica_count=1, model_display_name='my-trained-model', model_labels={'key': 'value'}, )
