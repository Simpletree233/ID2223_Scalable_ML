# ID2223_Scalable_ML
ID2223 Lab


Lab1 description:
[Source code](https://github.com/ID2223KTH/id2223kth.github.io/tree/master/src/serverless-ml-intro)

In this lab assignment, you will practice building a scalable serverless machine learning system, consisting of a feature pipeline, a training pipeline, a batch inference pipeline, and a user interface (one for interactive querying, one as a dashboard).

In the first task, you will build and run the Iris Flower Dataset as a serverless system. 

Go to dir and install the requiements using: `pip install -r requirements.txt`

In the second task, you will build a similar serverless ML service for the Titantic passenger survival dataset. You need to write the source code for this task yourself. It is a good idea to use the Iris Flower Dataset source code as a basis for building the Titantic service. 

## Task 1
Serverless ML

First Steps
1. Create a free account on hopsworks.ai
2. Create a free account on modal.com
2. Create a free account on huggingface.com

● Tasks
1. Build and run a feature pipeline on Modal
1. Build and run a training pipeline on Modal
1. Build and run an inference pipeline with a Gradio UI on Hugging Face 
Spaces.


## Task 2
The Titanic Dataset:
a. https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv

2. Write a feature pipeline that registers the titantic dataset as a Feature Group with 
Hopsworks. You are free to drop or clean up features with missing values.
3. Write a training pipeline that reads training data with a Feature View from Hopsworks, 
trains a binary classifier model to predict if a particular passenger survived the Titanic 
or not. Register the model with Hopsworks.
4. Write a Gradio application that downloads your model from Hopsworks and provides a 
User Interface to allow users to enter or select feature values to predict if a passenger 
with the provided features would survive or not.
5. Write a synthetic data passenger generator and update your feature pipeline to allow it 
to add new synthetic passengers.
6. Write a batch inference pipeline to predict if the synthetic passengers survived or not, 
and build a Gradio application to show the most recent synthetic passenger prediction 
and outcome, and a confusion matrix with historical prediction performance. 


References: https://www.kaggle.com/competitions/titanic/data

 https://www.ritchieng.com/pandas-scikit-learn/
