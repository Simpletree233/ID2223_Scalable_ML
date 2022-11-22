# ID2223_Scalable_ML_Lab1

Lab1 description:

## Task 1

Link to Iris prediction interface:
https://huggingface.co/spaces/Yuyang2022/Iris_Prediction

Link to Iris prediction monitor:
https://huggingface.co/spaces/Yuyang2022/Iris_Monitor

## Task 2

Part 1: Feature Engineering
Refer to *Machine_Learning_Model.ipynb* for the full implementation
1. Drop columns with low predictive power (For example, drop "Name", "Ticket", "PassengerId")
2. Convert Cabin column to Deck column
3. Impute Nan values (Age with median age, Deck,Embarked with "N" to indicate missing)
4. One hot encode categorical variables

Part 2: Model training
1. Selected Ensemble method Bagging for model
2. Use GridSearch to search for best parameter from specified parameter space
3. Use best parameter found to train model in titanic-training-pipeline.py

Part 3: Scripts to upload data to hopsworks
1. Uploaded transformed titanic dataset to hopsworks feature group (titanic-feature-pipeline.py)
2. Uploaded trained model to hopsworks model registry (titanic-training-pipeline.py)
3. Generate random features and upload to hopsworks (titanic-feature-pipeline-daily.py)
4. Predict and upload the results in the form of images to hopsworks (titanic-batch-inference-pipeline.py)
   - Image of today's actual label and prediction
 
   - Dataframe of latest 5 predictions
 
   - Confusion matrix of all previous predictions

Part 4: Hugging face interface

Link to titanic prediction interface
https://huggingface.co/spaces/WayneLinn/ID2223-Lab1-Titanic-Survivor-Prediction

> ### Instructions on how to use interface
> All entries must be provided to avoid errors. 
> Output interpretation (In case it is not clear)
> "Dead 110 years ago rip" -> dead
> "Dead but not on the titanic" -> survived

Link to titanic prediction monitor
https://huggingface.co/spaces/WayneLinn/ID2223-Lab1-Titanic-Prediction-Monitor

Bonus: Added 2 buttons to allow retrivial of the latest result and update the images.

> ### Instruction on how to use monitor
> 1. Press Load
> 2. Wait 2-5 seconds
> 3. Press refresh to see latest result


