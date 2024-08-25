import os

import boto3

AWS_SAGEMAKER_PIPELINE_NAME = os.environ["AWS_SAGEMAKER_PIPELINE_NAME"]
AWS_S3_MODELS_BUCKET_NAME = os.environ["AWS_S3_MODELS_BUCKET_NAME"]

sagemaker_client = boto3.client("sagemaker")


def lambda_handler(event, context):
    return sagemaker_client.start_pipeline_execution(
        PipelineName=AWS_SAGEMAKER_PIPELINE_NAME,
        PipelineExecutionDisplayName="DailyExecution",
        PipelineParameters=[
            {"Name": "AWS_S3_MODELS_BUCKET_NAME", "Value": AWS_S3_MODELS_BUCKET_NAME},
        ],
    )
