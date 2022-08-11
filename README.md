# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

-The model used is the Resnet50 image classification model. I choosed this model because ResNet-50 is 50 layers deep and is trained on a million images of 1000 categories from the ImageNet database. Furthermore the model has over 23 million trainable parameters, which indicates a deep architecture that makes it better for image recognition
-The hyperparameters chosen were, batch_size->CategoricalParameter([32, 64]), lr->ContinuousParameter(0.01, 0.1), epochs->IntegerParameter(5, 10), and momentum->ContinuousParameter(0.5, 0.9)

Remember that your README should:
- Include a screenshot of completed training jobs
![Training Jobs](training_jobs.png)

- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
I used this documntation https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-configure-framework-profiling.html
rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
]

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500, framework_profile_params=FrameworkProfile(num_steps=10)
)

debugger_config = DebuggerHookConfig(
    hook_parameters={"train.save_interval": "100", "eval.save_interval": "10"}
)

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
The debugging output has detected vanishing gradients, overfit and poor weight initialization issues.

**TODO** Remember to provide the profiler html/pdf file in your submission.
Done


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

pytorch_model = PyTorchModel(model_data=model_data, role=role, entry_point='inference.py', py_version="py36", framework_version="1.8")
predictor1 = pytorch_model.deploy(initial_instance_count=1, instance_type="ml.t2.medium")

-One can give path to the image he wants to predict and the endpoint will give inference to classify the dog breed and show the dog image.


**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![Endpoint](endpoints.png)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
