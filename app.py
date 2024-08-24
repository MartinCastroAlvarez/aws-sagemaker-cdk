import aws_cdk as core
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_apigateway as apigateway

import os
from typing import Dict

from aws_cdk import App
from aws_cdk import Environment


class SagemakerStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # S3 Bucket for model artifacts
        bucket = s3.Bucket(self, "ModelBucket")

        # IAM Role for SageMaker to access S3 Bucket
        sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")],
        )

        # Placeholder for SageMaker Model definitions
        # Assuming model artifacts are already in S3 and using a dummy image URI
        model_v1 = sagemaker.CfnModel(
            self,
            "ModelV1",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image="763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.3.0-cpu",
                model_data_url=f"s3://{bucket.bucket_name}/path/to/model_v1.tar.gz",
            ),
        )

        model_v2 = sagemaker.CfnModel(
            self,
            "ModelV2",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image="763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.3.0-cpu",
                model_data_url=f"s3://{bucket.bucket_name}/path/to/model_v2.tar.gz",
            ),
        )

        # Lambda Function to invoke SageMaker endpoints
        lambda_function = lambda_.Function(
            self,
            "LambdaFunction",
            runtime=lambda_.Runtime.PYTHON_3_8,
            handler="lambda_function.handler",
            code=lambda_.Code.from_asset(os.path.join(os.getcwd(), "lambda")),
        )

        # API Gateway to expose the Lambda function
        api = apigateway.LambdaRestApi(self, "EndpointAPI", handler=lambda_function, proxy=False)

        api.root.add_method("GET")  # Add GET method

        # Output the API URL
        core.CfnOutput(self, "APIURL", value=api.url)


if __name__ == "__main__":
    app: App = App()
    SagemakerStack(
        app,
        "SagemakerStack",
        profile=app.node.try_get_context("profile"),
        env=Environment(
            account=os.environ["CDK_DEFAULT_ACCOUNT"],
            region=os.environ["CDK_DEFAULT_REGION"],
        ),
    )
    app.synth()
