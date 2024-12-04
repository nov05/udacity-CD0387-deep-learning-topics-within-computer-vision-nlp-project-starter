# **Image Classification using AWS SageMaker**

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either [the provided dog breed classication data set](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available.   

  * Check [the Google Docs notes](https://docs.google.com/document/d/1OvnsKYyGk-ww8NVl7hdQxXkUAjU8WvIuTo9pzTLv2Ls/).


## 👉 Dataset
The provided dataset is [the dogbreed classification dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

* Simple data exploration ([notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/00.EDA.ipynb))  
  * Download two pictures to local and display them  
  * There are **133 classes** (133 dog breeds) and there is data imbalance. In the [`scripts/train.py`](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/scripts/train.py) file, `class weights` are added to the loss function. And two test runs show that it improves the accuracy from 74% to 79% for `ResNet50`.  
    ```python
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_dataset.targets), 
        y=train_dataset.targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  
    criterion = nn.CrossEntropyLoss(weight=class_weights)  
    ```

### 🏷️ Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
* Added `AmazonS3FullAccess` permission to the SageMaker execution role.  

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-02%2018_15_32-AmazonSageMaker-ExecutionRole-20241128T055392%20_%20IAM%20_%20Global.jpg" width=600>  

* A bucket `p3-dog-breed-classification` is created to store the project data.  
    * Folder `dogImages/` stores all the dog pictures
    * Folder `jobs/` stores all the job outputs  

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-02%2018_20_10-p3-dog-breed-classification%20-%20S3%20bucket%20_%20S3%20_%20us-east-1.jpg" width=600>  

## 👉 Hyperparameter Tuning  

🏷️ Check [the notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/02.train_and_deploy.ipynb)  

🏷️ What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

* The train/val/test dataset contains **8,351 images**, which, while not very large, is sufficient for fine-tuning a pre-trained model. We want the model to be deep enough to capture complex patterns but not so deep that it becomes overly computationally expensive. [The baseline results](https://wandb.ai/nov05/udacity-awsmle-resnet50-dog-breeds/reports/ResNet50-101-152-Baselines-on-Dog-Breed-Classification--VmlldzoxMDQzNDI2Nw) show that the performance of `ResNet-50` ([notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/01.simple_train_resnet50.ipynb)), `ResNet-101` ([notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/01.simple_train_resnet101.ipynb)), and `ResNet-152` ([notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/01.simple_train_resnet152.ipynb)) are quite similar. Therefore, we chose ResNet-50, as it strikes a good balance between computational efficiency and model depth, making it more practical for both training and deployment.

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-02%2022_50_56-ResNet50_101_152%20Baselines%20on%20Dog%20Breed%20Classification%20_%20udacity-awsmle-resnet50.jpg" width=600>

🏷️ Remember that your README should:   

- Include a screenshot of completed training jobs

  * An HPO job with 20 training runs is currently in progress.   

    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/20241203_aws%20sagemaker%20hpo%20jobs.jpg" width=600>

- Logs metrics during the training process  

  * For this project, I'm using `eval_loss_epoch`, which represents the average loss per step within an epoch.  

- Tune at least two hyperparameters
  * Unfortunately, there is a strict limit on **GPU instances** for training jobs. For example, only 2 training jobs can be created at a time for an HPO job using `ml.g4dn.xlarg` (Go to `Service Quotas > AWS services > Amazon SageMaker`, search for `ml.g4dn.xlarg`.) So, while I can demonstrate creating an HPO job, the results may not be optimal.  

    ```python
    hyperparameter_ranges = {
      'epochs': IntegerParameter(20, 40, scaling_type="Auto"),
      'batch-size': CategoricalParameter([16, 32, 64]),
      'opt-learning-rate': ContinuousParameter(1e-5, 1e-4),
      'opt-weight-decay': ContinuousParameter(1e-5, 1e-3),
    }
    objective_metric_name = "eval_loss_epoch"
    objective_type = "Minimize"
    metric_definitions = [{
        "Regex": "EVAL: Average loss: ([0-9\\.]+)"
    }]
    ```  

    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-03%2002_03_35-Quotas%20list%20-%20Amazon%20SageMaker%20_%20AWS%20Service%20Quotas.jpg" width=600>

- Retrieve the best hyperparameters from all your training jobs   

  ```python
  from sagemaker.estimator import Estimator
  from sagemaker.tuner import HyperparameterTuningJobAnalytics
  tuning_job_name = "p3-dog-breeds-hpo-241203-0321"
  hpo_analytics = HyperparameterTuningJobAnalytics(tuning_job_name, session)
  best_training_job = hpo_analytics.best_training_job()
  print("👉 Best training job:", best_training_job)
  best_estimator = Estimator.attach(best_training_job)
  print("👉 Best estimator hyperparameters:")
  best_estimator().hyperparameters()
  ```   

🏷️ **W&B Sweep**  

* W&B Sweep is somewhat similar to AWS SageMaker HPO, but W&B stands out by offering intuitive visual tools that help us understand how different hyperparameters impact training metrics. For instance, from the screenshots, we can infer that the optimizer's learning rate is likely the most significant hyperparameter, with an optimal value around 8e-5. Additionally, it seems best to keep the optimizer's weight decay very small.

  * [Check the Sweep workspace](https://wandb.ai/nov05/udacity-awsmle-resnet50-dog-breeds/sweeps/tkeo613o)    

    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-03%2019_24_26-sagemaker-hpo%20_%20udacity-awsmle-resnet50-dog-breeds%20Workspace%20%E2%80%93%20Weights%20%26%20Biases.jpg" width=600>  

    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-03%2019_51_24-sagemaker-hpo%20_%20udacity-awsmle-resnet50-dog-breeds%20Workspace%20%E2%80%93%20Weights%20%26%20Biases.jpg" width=600>  

## 👉 Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### 🏷️ Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## 👉 Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
