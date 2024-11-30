from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role


role_arn = get_execution_role()  ## get role ARN
if 'AmazonSageMaker-ExecutionRole' not in role_arn:
    ## your own role here
    role_arn = "arn:aws:iam::061096721307:role/service-role/AmazonSageMaker-ExecutionRole-20241128T055392"
print("👉 Role ARN:", role_arn) ## If local, Role ARN: arn:aws:iam::807711953667:role/voclabs

# TODO: Include the hyperparameters your script will need over here.
hyperparameters = {"epochs": "2", "batch-size": "32", "test-batch-size": "100", "lr": "0.001"}
# TODO: Create your estimator here. You can use Pytorch or any other framework.
estimator = PyTorch(
    entry_point="scripts/pytorch_cifar.py",
    base_job_name="sagemaker-script-mode",
    role=get_execution_role(),
    instance_count=1,
    instance_type="ml.m5.large",
    hyperparameters=hyperparameters,
    framework_version="1.8",
    py_version="py36",
)
# TODO: Start Training
estimator.fit(wait=True)