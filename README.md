# **🟢 P3 Submission: Dog Breed Image Classification using AWS SageMaker**  

This is the project folder for Project 3 `Dog Breed Image Classification`, Course 5 `Operationalizing Machine Learning on SageMaker`, [Udacity **AWS Machine Learning Engineer Nanodegree**](https://www.udacity.com/course/aws-machine-learning-engineer-nanodegree--nd189) (ND189). 

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either [the provided dog breed classication data set](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) or one of your choice.

* **Project overview**:  
  * There are **133 dog breeds** in the dataset. 
  * Finetuned a **pretrained ResNet50** model.
  * Reaching a **test accuracy of 91.39%** after 16 epochs (43 minutes) on an `ml.g4dn.xlarge` GPU instance 
  * Hyperparameter optimization, logging, debugging and profiling  
  * Deployed on a `ml.m5.large` CPU instance, endpoint tested by randomly chosen pictures from the test dataset in S3  
  * Result demo video (click to watch it on YouTube)  
    [<img src="https://github.com/nov05/pictures/blob/master/Udacity/20241119_aws-mle-nanodegree/p3%20dog%20breed%20classification%20resnet50.gif?raw=true" width=600>](https://www.youtube.com/watch?v=JWUoDsbUt5o)  

* **Submission artifacts**:  
  All files are in the repo root directory:
  * train_and_deploy.ipynb ([this file](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/02.train_hpo_debug_deploy.ipynb))
  * train.py ([this file](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/scripts/train.py), and [inference.py](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/deploy_scripts/inference.py))
  * hpo.py (for hyperparameter optimization, same with `train.py`)
  * PDF/HTML of the Profiling Report ([this file](https://nov05.github.io/htmls/udacity/20241207_aws-mle-nanodegree/profiler-report.html))
  * README.md (what you are reading now)

<br>  

## **👉 Project Set Up and Installation**     

Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available.   

  * Check [the Google Docs notes](https://docs.google.com/document/d/1OvnsKYyGk-ww8NVl7hdQxXkUAjU8WvIuTo9pzTLv2Ls/).  

  * For this project, I used `SageMaker` locally in `VS Code` for a better IDE experience. However, the debugging and profiling reports were generated in `AWS SageMaker Studio`, as the libraries didn’t seem to work properly in the local environment (`Windows OS`).     

    * Set up [the local conda env `awsmle_py310`](https://gist.github.com/nov05/d9c3be6c2ab9f6c050e3d988830db08b) (without `CUDA`, as all the jobs were executed on `AWS`)     

  * The major training and deployment scripts are organized in the following code structure:    
    ```text
    repo/  
    │  
    ├── 02.train_hpo_debug_deploy.ipynb   
    ├── scripts/  
    │   ├── train.py  
    │   └── requirements.txt    
    └── deploy_scripts/   
        └── inference.py       
    ```

<br>

## **👉 Dataset**  

The provided dataset is [the dogbreed classification dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) which can be found in the classroom.  

~~The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.~~

* Simple data exploration ([notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/00.EDA.ipynb))   

  * There are **a total of 8,351 images**, already divided into train, validation, and test sets: 6,680 for training, 835 for validation, and 836 for testing. While this dataset may not be large enough to train a deep learning model from scratch, it should be sufficient for fine-tuning a pretrained model, like ResNet50. 

    So far, the best test accuracy I've achieved is **91.39%**, which indicates that adding more training data could help improve performance. However, since I don't have the time to do that, we'll stick with the current dataset for this project.

  * Download two images locally and display them to get a better feel for the data.      

  * There are **133 classes** (133 dog breeds) and there is data imbalance. In the [`scripts/train.py`](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/scripts/train.py) file, `class weights` are added to the loss function. And two test runs show that it improves the accuracy from 74% to 79% for `ResNet50`.  
    ```python
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_dataset.targets), 
        y=train_dataset.targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  
    criterion = nn.CrossEntropyLoss(weight=class_weights)  
    ```

<br>  

### 🏷️ Access  

Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
* Added `AmazonS3FullAccess` permission to the SageMaker execution role.  

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-02%2018_15_32-AmazonSageMaker-ExecutionRole-20241128T055392%20_%20IAM%20_%20Global.jpg" width=600>  

* A bucket `p3-dog-breed-classification` is created to store the project data.  
    * Folder `dogImages/` stores all the dog pictures
    * Folder `jobs/` stores all the job outputs  

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-02%2018_20_10-p3-dog-breed-classification%20-%20S3%20bucket%20_%20S3%20_%20us-east-1.jpg" width=600>  

<br>  

## **👉 Hyperparameter Tuning**  

🏷️ Check [02.train_hpo_debug_deploy.ipynb](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/02.train_hpo_debug_deploy.ipynb)  

<br>  

🏷️ What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

* The train/val/test dataset contains **8,351 images**, which, while not very large, is sufficient for fine-tuning a pre-trained model. We want the model to be deep enough to capture complex patterns but not so deep that it becomes overly computationally expensive. [The baseline results](https://wandb.ai/nov05/udacity-awsmle-resnet50-dog-breeds/reports/ResNet50-101-152-Baselines-on-Dog-Breed-Classification--VmlldzoxMDQzNDI2Nw) show that the performance of `ResNet-50` ([notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/01.simple_train_resnet50.ipynb)), `ResNet-101` ([notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/01.simple_train_resnet101.ipynb)), and `ResNet-152` ([notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/01.simple_train_resnet152.ipynb)) are quite similar. Therefore, we chose ResNet-50, as it strikes a good balance between computational efficiency and model depth, making it more practical for both training and deployment.

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-02%2022_50_56-ResNet50_101_152%20Baselines%20on%20Dog%20Breed%20Classification%20_%20udacity-awsmle-resnet50.jpg" width=600>

<br>  

🏷️ Remember that your README should:   

- Include a screenshot of completed training jobs

  * An HPO job with 20 training runs is currently in progress.   

    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/20241203_aws%20sagemaker%20hpo%20jobs.jpg" width=600>

- Logs metrics during the training process  

  * For this project, I'm using `eval_loss_epoch`, which represents the average loss per step within an epoch.  

- Tune at least two hyperparameters
  * Unfortunately, there is a strict limit on **GPU instances** for training jobs. For example, only 2 training jobs can be created at a time for an HPO job using `ml.g4dn.xlarge` (Go to `Service Quotas > AWS services > Amazon SageMaker`, search for `ml.g4dn.xlarge`.) So, while I can demonstrate creating an HPO job, the results may not be optimal.  

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
    from pprint import pprint
    tuning_job_name = "p3-dog-breeds-hpo-241203-0321"
    hpo_analytics = HyperparameterTuningJobAnalytics(tuning_job_name, session)
    df_tuning_results = hpo_analytics.dataframe()
    best_training_job = df_tuning_results.sort_values('FinalObjectiveValue', ascending=True).iloc[0]
    print("👉 Best training job hyperparameters:")
    best_training_job
    ```   
    ```text
    👉 Best training job hyperparameters:
    batch-size                                                          "32"
    epochs                                                              20.0
    opt-learning-rate                                                0.00008
    opt-weight-decay                                                0.000025
    TrainingJobName               p3-dog-breeds-hpo-241203-0321-018-c70921b4
    TrainingJobStatus                                              Completed
    FinalObjectiveValue                                               0.8678
    TrainingStartTime                              2024-12-03 10:04:43-06:00
    TrainingEndTime                                2024-12-03 10:50:50-06:00
    TrainingElapsedTimeSeconds                                        2767.0
    ```


* The project budget is $25, so it's not ideal to run extensive hyperparameter optimization jobs on a GPU instance.  

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-03%2021_48_02-Amazon%20SageMaker%20AI%20_%20us-east-1.jpg.jpg" width=600>   

<br>  

* 🟢 Because of the AWS budget limit, the HPO job didn’t yield optimal results. So, I manually tuned the parameters and achieved a **test accuracy of 91.39%** in 16 epochs (43 minutes) on `ml.g4dn.xlarge`. You can [check the W&B logs for details](https://wandb.ai/nov05/udacity-awsmle-resnet50-dog-breeds/runs/p3-dog-breeds-job-20241207-112640-958tti-algo-1?nw=nwusernov05). I used the following hyperparameters, implemented **early stopping** (if the evaluation loss didn’t decrease for 5 epochs), and added a **learning rate scheduler** that reduced the optimizer’s LR by half every 5 epochs (`torch.optim.AdamW` and `torch.optim.lr_scheduler.StepLR`). 

  ```text 
  TrainingJobName: p3-dog-breeds-debug-20241204-124107 
  hyperparameters = {
      'epochs': 40,  ## trained 16 epochs
      'batch-size': 32,   
      'opt-learning-rate': 8e-5,  
      'opt-weight-decay': 1e-5,  
      'lr-sched-step-size': 5,  ## by epoch
      'lr-sched-gamma': 0.5,
      'early-stopping-patience': 5,
      'model-type': 'resnet50',  ## pre-trained
      'debug': True,  
  } 
  ```
  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-07%2012_10_48-Settings.jpg" width=600>  

<br>  

🏷️ **W&B Sweep**  

* W&B Sweep is somewhat similar to AWS SageMaker HPO, but W&B stands out by offering intuitive visual tools that help us understand how different hyperparameters impact training metrics. For instance, from the screenshots, we can infer that the optimizer's learning rate is likely the most significant hyperparameter, with an optimal value around 8e-5. Additionally, it seems best to keep the optimizer's weight decay very small.

  * [Check the Sweep workspace](https://wandb.ai/nov05/udacity-awsmle-resnet50-dog-breeds/sweeps/tkeo613o)    

    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-03%2019_24_26-sagemaker-hpo%20_%20udacity-awsmle-resnet50-dog-breeds%20Workspace%20%E2%80%93%20Weights%20%26%20Biases.jpg" width=600>  

    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-03%2019_51_24-sagemaker-hpo%20_%20udacity-awsmle-resnet50-dog-breeds%20Workspace%20%E2%80%93%20Weights%20%26%20Biases.jpg" width=600>  

<br>  

## **👉 Debugging and Profiling**  

* Give an overview of how you performed model debugging and profiling in Sagemaker

  * Define debugging and profiling rules for analyzing training issues before a training run.   

    ```python
    rules = [
        Rule.sagemaker(rule_configs.poor_weight_initialization()),
        Rule.sagemaker(rule_configs.vanishing_gradient()),
        Rule.sagemaker(rule_configs.overfit()),
        Rule.sagemaker(rule_configs.overtraining()),
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
        ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
        ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    ]
    hook_config = DebuggerHookConfig(
        hook_parameters={
            "train.save_interval": "100", 
            "eval.save_interval": "10"
        }
    )
    ```

  * SageMaker creats processing jobs according to the rules. In my case, there is a limit of 10 jobs that can run simultaneously at the account level.   
    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-07%2003_42_35-Amazon%20SageMaker%20AI%20_%20us-east-1.jpg" width=600>  

  * Once training job is complete, SageMaker automatically generates detailed debugging and profiling reports, which you can view in the AWS Management Console or download for further analysis. These reports highlight potential issues, such as high latency or under-utilization of resources.  

<br>  

### 🏷️ Results  

  * What are the results/insights did you get by profiling/debugging your model?

    * ⚠️ E.g. Got an warning message when debugging: `PoorWeightInitialization: IssuesFound`   
      ```text
      [12/04/24 02:10:21] WARNING  Job ended with status 'Stopped' rather than 'Completed'. This could    session.py:8593
                                    mean the job timed out or stopped early for some other reason:                        
                                    Consider checking whether it completed as you expect.   
      ```
      **Solution**: Initiate the fc layer explicitly.  
      ```python
      torch.nn.init.kaiming_normal_(model.fc.weight)  # Initialize new layers
      ```

    * With 90% of the time spent in the `Training Loop` and `GPU utilization` at 36%, both of those seem reasonable.  

    * ⚠️ The `GPUMemoryIncrease` rule was triggered 734 times, indicating that the memory footprint is nearing the available limit. It suggests opting for a larger instance type with more memory. I went with a smaller GPU instance to save on costs.    

    * ⚠️ The `LowGPUUtilization` rule was triggered 35 times, suggesting there might be bottlenecks and recommending actions like minimizing blocking calls, changing the distributed training strategy, or increasing the batch size. I set the batch size to 32, but it could likely be increased.

  * Remember to provide the profiler html/pdf file in your submission.

    * [View the html report](https://nov05.github.io/htmls/udacity/20241207_aws-mle-nanodegree/profiler-report.html) (hosted on Github)  

      [<img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-07%2020_38_11-profiler-report.jpg" width=600>](https://nov05.github.io/htmls/udacity/20241207_aws-mle-nanodegree/profiler-report.html)  

    * Check [the debug job notebook](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/03.debug_trial_profile.ipynb)  

<br>  

## **👉 Model Deployment**  

* Give an overview of the deployed model and instructions on how to query the endpoint with a sample input. 

  * Check the [`deploy_scrpits/inference.py`](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/deploy_scripts/inference.py) file. There are 4 custom functions for inference endpoint to handle the models loading/prediction and data input/output: `model_fn()`, `input_fn()`, `predict_fn()`, `output_fn()`.    

  * I decided to attach a `JSON serializer` and `JSON deserializer` to the estimator. (I tried using the Identity serializer for image data in my last course project.)  

    * Input data needs to be encoded probably before sending to the endpoint.  
      *P.S. The following code works, but JSON dumps and loads are probably unnecessary since the JSON serializer is already attached. I'll test it out and update the code accordingly.*      
      ```python
      with open(local_object, "rb") as f:
        payload = json.dumps(
            base64.b64encode(f.read()).decode('utf-8')
        )
      response = predictor.predict(payload)
      ```
    * In the `input_fn()`, input data needs to be decoded accordingly.   

      ```python
      data_bytes = base64.b64decode(
        json.loads(input_data)
      )
      data_image = Image.open(BytesIO(data_bytes)).convert('RGB')  # Ensure 3-channel RGB
      ```

    * It is simpler for the output data. Here `prediction` in the `output_fn()` is a list.

      ```python
      return json.dumps(prediction)
      ```

    * And the response of the predictor is a list. 

      ```python
      response = predictor.predict(payload)
      ```

  * You can either deploy an endpoint **from a training job**.   

    ```python
    training_job_name = "p3-dog-breeds-debug-20241204-124107" ## best test accuracy
    estimator = PyTorch.attach(training_job_name)
    predictor=estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large", ## "ml.m5.xlarge", "ml.g4dn.xlarge"
        endpoint_name="p3-dog-breed-classification",  ## ⚠️ naming conventions
        entry_point="inference.py",  # Reference to your inference.py
        source_dir="deploy_scripts"         # Directory containing inference.py
    )
    ```

  * Or deploy an endpoint **from a model artifact** (e.g. state dict, or `TorchScript` saved model).    

    ```python  
    model_s3_uri = r"s3://p3-dog-breed-image-classification/p3-dog-breeds-debug-20241204-124107/output/model.tar.gz"  
    pytorch_model = PyTorchModel(
        model_data=model_s3_uri,   # Path to the S3 model file
        role=role_arn,            
        framework_version='1.10',    # Adjust the PyTorch version as needed
        py_version='py38',           # Python version (adjust if necessary)
        entry_point='inference.py',  # Inference script for loading and predicting (see details below)
        source_dir="deploy_scripts",         # Directory containing inference.py
    )
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',  # You can adjust the instance type based on your needs
        endpoint_name='p3-dog-breed-classification',
    )
    ```

* Remember to provide a screenshot of the deployed active endpoint in Sagemaker.  

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/p3-dog-breed-classification_endpoint.jpg" width=600>  


<br>  

## **⚠️ Clean Up Resources (IMPORTANT)**   

* Delete endpoint model and endpoint   

  ```python
  import boto3
  ## delete the endpoint model
  sm_client = boto3.client('sagemaker')
  endpoint_info = sm_client.describe_endpoint(EndpointName=predictor.endpoint_name)
  endpoint_config_name = endpoint_info['EndpointConfigName']
  endpoint_config_info = sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
  model_name = endpoint_config_info['ProductionVariants'][0]['ModelName']
  sm_client.delete_model(ModelName=model_name)
  print("🟢 Model deleted:", model_name)
  ## delete the endpoint
  predictor.delete_endpoint()
  print("🟢 Endpoint deleted:", predictor.endpoint_name)
  ```

* Delete job outputs in an S3 bucket if no longer needed
  ```python
  ## clear the sagemaker default bucket where stores the deployable model created by the endpoint
  !aws s3 rm s3://sagemaker-us-east-1-852125600954 --recursive
  ```

* Delete endpoint log streams 
  ```python 
  import boto3
  def delete_log_streams(s3_client_logs, log_group_name):
      # Retrieve log streams from the specified log group
      log_streams = s3_client_logs.describe_log_streams(logGroupName=log_group_name)
      # Iterate through each log stream and delete it
      for stream in log_streams['logStreams']:
          log_stream_name = stream['logStreamName']
          print(f"Deleting log stream: {log_stream_name}...")
          s3_client_logs.delete_log_stream(logGroupName=log_group_name, logStreamName=log_stream_name)
      print(f"🟢 All log streams deleted from log group: {log_group_name}")

  s3_client_logs = boto3.client('logs')
  log_group_name = "/aws/sagemaker/Endpoints/p3-dog-breed-classification"
  delete_log_streams(s3_client_logs, log_group_name)  
  ```

* After you're done working in the AWS `SageMaker Studio` console, make sure there are **no running instances**, and delete the SageMaker Studio space if you no longer need it, as it can incur costs even when it's not in use.   

<br>  

## 🟢 Standout Suggestions  

**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.  

* [W&B logging and sweeping](https://wandb.ai/) are incorporated to gain better insights into the importance of hyperparameters and to explore their potential range.    

  * In the training notebook, the following code generates wandb credential in the `scripts/` dir.    
    ```python
    import wandb
    ## generate secrets.env. remember to add it to .gitignore  
    wandb.sagemaker_auth(path="scripts")  
    ```

  * In the `scripts/train.py` file, the following code initiate and finish a wandb run.   
    ```python
    wandb.init(
        ## set the wandb project where this run will be logged
        project="udacity-awsmle-resnet50-dog-breeds",
        config=vars(parser.parse_args())
    )
    ## your code inbetween
    wandb.finish()
    ```
  * In the `scripts/train.py` file, the following code logs training loss. You can always log other metrics.     
    ```python
    wandb.log({"train_loss": loss.item()}, step=step_counter.total_steps)
    ```

* All operations, except for generating debugging and profiling reports, are executed locally using Python code and `AWS CLI`. This significantly improved workflow automation and reduced the costs associated with `SageMaker Studio`.   
  * [Medium post](https://medium.com/@liuwenjing.datascience/quick-tips-for-minimizing-aws-sagemaker-costs-in-machine-learning-exercises-874e25a4d23b)  

* Resources are cleaned up after tasks are completed to minimize costs.    

* If I had more time, I would explore distributed training, asynchronous inference, batch transformation, deploying multiple models with production variants, and using SageMaker Clarify, among other features.      

* Checking [the official SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) is a good practice.   

  * `Model training > Debugging and improving model performance`  

    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-06%2017_02_03-Settings-min.jpg" width=600>   