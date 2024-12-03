# **Image Classification using AWS SageMaker**

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either [the provided dog breed classication data set](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

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
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

* The train/val/test dataset contains **8,351 images**, which, while not very large, is sufficient for fine-tuning a pre-trained model. We want the model to be deep enough to capture complex patterns but not so deep that it becomes overly computationally expensive. [The baseline results](https://wandb.ai/nov05/udacity-awsmle-resnet50-dog-breeds/reports/ResNet50-101-152-Baselines-on-Dog-Breed-Classification--VmlldzoxMDQzNDI2Nw) show that the performance of `ResNet-50`, `ResNet-101`, and `ResNet-152` are quite similar. Therefore, we chose ResNet-50, as it strikes a good balance between computational efficiency and model depth, making it more practical for both training and deployment.

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2024-12-02%2022_50_56-ResNet50_101_152%20Baselines%20on%20Dog%20Breed%20Classification%20_%20udacity-awsmle-resnet50.jpg" width=600>

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best hyperparameters from all your training jobs

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
