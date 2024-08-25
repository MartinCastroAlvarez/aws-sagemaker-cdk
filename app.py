import json
import os
from dataclasses import dataclass
from typing import List

import aws_cdk as core
import boto3
from aws_cdk import App
from aws_cdk import Environment
from aws_cdk import aws_apigateway
from aws_cdk import aws_ecr
from aws_cdk import aws_events
from aws_cdk import aws_events_targets
from aws_cdk import aws_glue as glue
from aws_cdk import aws_iam
from aws_cdk import aws_lambda
from aws_cdk import aws_s3
from aws_cdk import aws_sagemaker

s3_client = boto3.client("s3")
current_directory = os.path.dirname(__file__)


@dataclass
class Version:
    name: str
    version: str


def get_versions(bucket_name: str) -> List[Version]:
    return [
        Version(
            name=item["Key"].split("/")[0],
            version=item["Key"].split("/")[1],
        )
        for item in s3_client.list_objects_v2(Bucket=bucket_name)["Contents"]
        if item["Key"].endswith("model.pth")
    ]


class RegistryStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, environment_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)
        repository = aws_ecr.Repository(
            self, f"ecr-repository-{environment_name}", repository_name=f"{environment_name}-repository"
        )
        core.CfnOutput(
            self,
            f"ecr-repository-uri-{environment_name}",
            export_name=f"ecr-repository-uri-{environment_name}",
            value=repository.repository_uri,
        )


class BucketsStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, environment_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)
        models_bucket = aws_s3.Bucket(self, f"s3-{environment_name}-models-bucket")
        core.CfnOutput(
            self,
            f"s3-models-bucket-name-{environment_name}",
            export_name=f"s3-models-bucket-name-{environment_name}",
            value=models_bucket.bucket_name,
        )
        features_bucket = aws_s3.Bucket(self, f"s3-{environment_name}-features-bucket")
        core.CfnOutput(
            self,
            f"s3-features-bucket-name-{environment_name}",
            export_name=f"s3-features-bucket-name-{environment_name}",
            value=features_bucket.bucket_name,
        )


class GlueDatabaseStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, environment_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)
        db = glue.Database(self, f"glue-database-{environment_name}", database_name=f"glue_database_{environment_name}")
        core.CfnOutput(
            self,
            f"glue-database-name-{environment_name}",
            value=db.database_name,
            export_name=f"glue-database-name-{environment_name}",
        )


class FeatureStoreStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, environment_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)
        features_bucket = aws_s3.Bucket.from_bucket_name(
            self,
            f"s3-features-bucket-name-{environment_name}",
            bucket_name=core.Fn.import_value(f"s3-features-bucket-name-{environment_name}"),
        )
        glue_database: glue.Database = glue.Database.from_database_name(
            self,
            f"glue-database-{environment_name}",
            database_name=core.Fn.import_value(f"glue-database-name-{environment_name}"),
        )
        self.feature_group = aws_sagemaker.CfnFeatureGroup(
            self,
            f"feature-group-{environment_name}",
            feature_group_name=f"features-{environment_name}",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=[
                {"FeatureName": "feature1", "FeatureType": "Integral"},
                {"FeatureName": "feature2", "FeatureType": "Fractional"},
                {"FeatureName": "feature3", "FeatureType": "String"},
            ],
            offline_store_config={
                "S3StorageConfig": {"S3Uri": f"s3://{features_bucket.bucket_name}/feature-store/"},
                "DataCatalogConfig": {
                    "TableName": f"feature_store_table_{environment_name}",
                    "Catalog": "AwsDataCatalog",
                    "Database": glue_database.database_name,  # Enables Glue Data Catalog for Athena.
                },
            },
        )
        core.CfnOutput(self, f"feature-group-name-{environment_name}", value=self.feature_group.feature_group_name)


class PipelinesStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, environment_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)
        models_bucket = aws_s3.Bucket.from_bucket_name(
            self,
            f"s3-models-bucket-name-{environment_name}",
            bucket_name=core.Fn.import_value(f"s3-models-bucket-name-{environment_name}"),
        )
        repository = aws_ecr.Repository.from_repository_name(
            self,
            f"ecr-repository-{environment_name}",
            repository_name=f"{environment_name}-repository",
        )
        role = aws_iam.Role(
            self,
            f"sagemaker-pipeline-role-{environment_name}",
            assumed_by=aws_iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )
        role.attach_inline_policy(
            aws_iam.Policy(
                self,
                f"role-s3-access-policy-{environment_name}",
                statements=[
                    aws_iam.PolicyStatement(
                        actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                        resources=[f"{models_bucket.bucket_arn}", f"{models_bucket.bucket_arn}/*"],
                    )
                ],
            )
        )
        role.attach_inline_policy(
            aws_iam.Policy(
                self,
                f"role-feature-store-access-policy-{environment_name}",
                statements=[
                    aws_iam.PolicyStatement(
                        actions=["sagemaker:GetRecord", "sagemaker:GetFeatureGroup", "sagemaker:BulkGetRecord"],
                        resources=[
                            f"arn:aws:sagemaker:{os.environ['CDK_DEFAULT_REGION']}:{os.environ['CDK_DEFAULT_ACCOUNT']}:feature-group/{environment_name}"
                        ],
                    )
                ],
            )
        )
        role.attach_inline_policy(
            aws_iam.Policy(
                self,
                f"role-ecr-access-policy-{environment_name}",
                statements=[
                    aws_iam.PolicyStatement(
                        actions=["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
                        resources=[repository.repository_arn],
                    )
                ],
            )
        )
        m1_v1_training_job = aws_sagemaker.CfnTrainingJob(
            self,
            f"training-job-{environment_name}",
            algorithm_specification=aws_sagemaker.CfnTrainingJob.AlgorithmSpecificationProperty(
                training_image=f"{repository.repository_uri}:m1-v1", training_input_mode="File"
            ),
            input_data_config=[
                aws_sagemaker.CfnTrainingJob.ChannelProperty(
                    channel_name="training",
                    data_source=aws_sagemaker.CfnTrainingJob.DataSourceProperty(
                        feature_group_data_source=aws_sagemaker.CfnTrainingJob.FeatureGroupDataSourceProperty(
                            feature_group_name=core.Fn.import_value(f"feature-group-name-{environment_name}")
                        )
                    ),
                )
            ],
            output_data_config=aws_sagemaker.CfnTrainingJob.OutputDataConfigProperty(
                s3_output_path=f"{models_bucket.bucket_arn}/m1/v1/"
            ),
            resource_config=aws_sagemaker.CfnTrainingJob.ResourceConfigProperty(
                instance_type="ml.m5.large", instance_count=1, volume_size_in_gb=50
            ),
            stopping_condition=aws_sagemaker.CfnTrainingJob.StoppingConditionProperty(max_runtime_in_seconds=360000),
            hyper_parameters={
                "epochs": "10",
                "lr": "0.01",
                "batch-size": "64",
                "limit": "100000000",
                "feature_group_name": core.Fn.import_value(f"feature-group-name-{environment_name}"),
            },
            environment={
                "AWS_DEFAULT_REGION": os.environ["CDK_DEFAULT_REGION"],
                "ENVIROMENT_NAME": environment_name,
            },
            role_arn=role.role_arn,
        )
        pipeline = aws_sagemaker.CfnPipeline(
            self,
            f"sagemaker-pipeline-{environment_name}",
            role_arn=role.role_arn,
            pipeline_name=f"{environment_name}-daily-pipeline",
            pipeline_definition=json.dumps(
                {
                    "PipelineDefinitionBody": {
                        "Version": "2020-12-01",
                        "Steps": [
                            {
                                "Name": "ExampleProcessingStep",
                                "Type": "Processing",
                                "Arguments": {},
                            },
                            m1_v1_training_job,
                        ],
                    }
                }
            ),
        )
        core.CfnOutput(
            self,
            f"pipeline-arn-{environment_name}",
            export_name=f"pipeline-arn-{environment_name}",
            value=pipeline.attr_arn,
        )
        trigger_lambda = aws_lambda.Function(
            self,
            f"trigger-lambda-{environment_name}-daily-pipeline",
            runtime=aws_lambda.Runtime.PYTHON_3_8,
            handler="trigger.lambda_handler",
            code=aws_lambda.Code.from_asset(os.path.join(current_directory, "lambdas", "triggers")),
            environment={
                "AWS_SAGEMAKER_PIPELINE_NAME": pipeline.pipeline_name,
                "AWS_S3_MODELS_BUCKET_NAME": models_bucket.bucket_name,  # Using bucket name in pipeline steps
            },
        )
        trigger_lambda.add_to_role_policy(
            aws_iam.PolicyStatement(
                actions=["sagemaker:StartPipelineExecution"],
                resources=[
                    pipeline.attr_arn,
                ],
            )
        )
        rule = aws_events.Rule(
            self,
            f"rule-{environment_name}",
            schedule=aws_events.Schedule.cron(minute="0", hour="0", month="*", year="*", week_day="*"),
            targets=[aws_events_targets.LambdaFunction(trigger_lambda)],
        )
        core.CfnOutput(
            self,
            f"rule-name-{environment_name}",
            export_name=f"rule-name-{environment_name}",
            value=rule.rule_name,
        )


class ModelsStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, environment_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)
        models_bucket = aws_s3.Bucket.from_bucket_name(
            self,
            f"s3-models-bucket-name-{environment_name}",
            bucket_name=core.Fn.import_value(f"s3-models-bucket-name-{environment_name}"),
        )
        role = aws_iam.Role(
            self,
            f"sagemaker-models-role-{environment_name}",
            assumed_by=aws_iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )
        role.attach_inline_policy(
            aws_iam.Policy(
                self,
                f"role-s3-access-policy-{environment_name}",
                statements=[
                    aws_iam.PolicyStatement(
                        actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                        resources=[models_bucket.bucket_arn, f"{models_bucket.bucket_arn}/*"],
                    )
                ],
            )
        )
        for version in get_versions(models_bucket.bucket_name):
            model = aws_sagemaker.CfnModel(
                self,
                f"model-{environment_name}-{version.name}-{version.version}",
                execution_role_arn=role.role_arn,
                primary_container=aws_sagemaker.CfnModel.ContainerDefinitionProperty(
                    image="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.6.0-cpu-py3",
                    model_data_url=f"s3://{models_bucket.bucket_name}/{version.name}/{version.version}/model.tar.gz",
                ),
            )
            config = aws_sagemaker.CfnEndpointConfig(
                self,
                f"config-{environment_name}-{version.name}",
                production_variants=[
                    aws_sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                        model_name=model.model_name,
                        variant_name=f"{version.name}-{version.version}",
                        initial_instance_count=1,
                        instance_type="ml.m5.large",
                    )
                ],
            )
            endpoint = aws_sagemaker.CfnEndpoint(
                self,
                f"endpoint-{environment_name}-{version.name}",
                endpoint_config_name=config.attr_endpoint_config_name,
            )
            core.CfnOutput(
                self,
                f"endpoint-name-{environment_name}-{version.name}",
                export_name=f"endpoint-name-{environment_name}-{version.name}",
                value=endpoint.attr_endpoint_name,
            )

    def get_versions(self, bucket_name: str) -> List[Version]:
        return [
            Version(
                name=item["Key"].split("/")[0],
                version=item["Key"].split("/")[1],
            )
            for item in s3_client.list_objects_v2(Bucket=bucket_name)["Contents"]
        ]


class ApiStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, environment_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)
        models_bucket = aws_s3.Bucket.from_bucket_name(
            self,
            f"s3-models-bucket-name-{environment_name}",
            bucket_name=core.Fn.import_value(f"s3-models-bucket-name-{environment_name}"),
        )
        api = aws_apigateway.RestApi(
            self,
            f"models-api-{environment_name}",
            rest_api_name=f"models-api-{environment_name}",
        )
        core.CfnOutput(self, f"api-url-{environment_name}", value=api.url)
        model_resource = api.root.add_resource("models")
        for version in get_versions(models_bucket.bucket_name):
            lambda_function = aws_lambda.Function(
                self,
                f"models-lambda-{environment_name}-{version.name}-{version.version}",
                runtime=aws_lambda.Runtime.PYTHON_3_8,
                handler="handler.handler",
                code=aws_lambda.Code.from_asset(os.path.join(current_directory, "lambdas", "models")),
                environment={
                    "AWS_SAGEMAKER_ENDPOINT_NAME": core.Fn.import_value(
                        f"endpoint-name-{environment_name}-{version.name}"
                    )
                },
            )
            model_resource.add_method(
                "POST",
                aws_apigateway.LambdaIntegration(lambda_function),
            )


if __name__ == "__main__":
    app: App = App()

    environment_name = app.node.try_get_context("environment_name")

    buckets_stack = BucketsStack(
        app,
        f"buckets-stack-{environment_name}",
        environment_name=environment_name,
        env=Environment(
            account=os.environ["CDK_DEFAULT_ACCOUNT"],
            region=os.environ["CDK_DEFAULT_REGION"],
        ),
    )

    registry_stack = RegistryStack(
        app,
        f"registry-stack-{environment_name}",
        environment_name=environment_name,
        env=Environment(
            account=os.environ["CDK_DEFAULT_ACCOUNT"],
            region=os.environ["CDK_DEFAULT_REGION"],
        ),
    )

    glue_stack = GlueDatabaseStack(
        app,
        f"glue-stack-{environment_name}",
        environment_name=environment_name,
        env=Environment(
            account=os.environ["CDK_DEFAULT_ACCOUNT"],
            region=os.environ["CDK_DEFAULT_REGION"],
        ),
    )

    features_stack = FeatureStoreStack(
        app,
        f"features-stack-{environment_name}",
        environment_name=environment_name,
        env=Environment(
            account=os.environ["CDK_DEFAULT_ACCOUNT"],
            region=os.environ["CDK_DEFAULT_REGION"],
        ),
    )
    features_stack.add_dependency(buckets_stack)
    features_stack.add_dependency(glue_stack)

    pipelines_stack = PipelinesStack(
        app,
        f"pipelines-stack-{environment_name}",
        environment_name=environment_name,
        env=Environment(
            account=os.environ["CDK_DEFAULT_ACCOUNT"],
            region=os.environ["CDK_DEFAULT_REGION"],
        ),
    )
    pipelines_stack.add_dependency(buckets_stack)
    pipelines_stack.add_dependency(features_stack)

    models_stack = ModelsStack(
        app,
        f"models-stack-{environment_name}",
        environment_name=environment_name,
        env=Environment(
            account=os.environ["CDK_DEFAULT_ACCOUNT"],
            region=os.environ["CDK_DEFAULT_REGION"],
        ),
    )
    models_stack.add_dependency(buckets_stack)

    api_stack = ApiStack(
        app,
        f"api-stack-{environment_name}",
        environment_name=environment_name,
        env=Environment(
            account=os.environ["CDK_DEFAULT_ACCOUNT"],
            region=os.environ["CDK_DEFAULT_REGION"],
        ),
    )

    app.synth()
