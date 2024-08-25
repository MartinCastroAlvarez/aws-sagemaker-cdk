import json
import os

import boto3

AWS_SAGEMAKER_ENDPOINT_NAME = os.environ["AWS_SAGEMAKER_ENDPOINT_NAME"]


def invoke_sagemaker_endpoint(data):
    runtime = boto3.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName=AWS_SAGEMAKER_ENDPOINT_NAME, ContentType="application/json", Body=json.dumps(data)
    )
    return json.loads(response["Body"].read().decode())


def handler(event, context):
    data = json.loads(event["body"])
    result = invoke_sagemaker_endpoint(data)
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(result)}
