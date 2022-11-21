import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub("titanic_prediction")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("lab1"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login(project="test42")
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_model_modal", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_feature_modal", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    #print(f"batch data:{batch_data}")
    #print(y_pred)
    state = y_pred[y_pred.size-1]
    survive=["dead","survived"]
    if survive[int(state)]=="dead":
        #dead
        state_url = "https://i.imgflip.com/5jvc2d.jpg"
    else:
        state_url="https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcTv3XuQKvjiF_ZpUt8rKlsVBX--JXMpfa674H03N-aApj_HjK1S"
    print("predicted: " + survive[state])
    img = Image.open(requests.get(state_url, stream=True).raw)            
    img.save("./latest_passenger.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_passenger.png", "Resources/images", overwrite=True)
    
    fg = fs.get_feature_group(name="titanic_feature_modal", version=1)
    df = fg.read()
    label = df.iloc[-1]["survived"]
    if survive[int(label)]=="dead":
        #dead
        label_url = "https://i.imgflip.com/5jvc2d.jpg"
    else:
        label_url="https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcTv3XuQKvjiF_ZpUt8rKlsVBX--JXMpfa674H03N-aApj_HjK1S"
    
    print("actual: " + survive[int(label)])
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_state.png")
    dataset_api.upload("./actual_state.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic survivor Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survive[state]],
        'label': [survive[int(label)]],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent_state.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_state.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different states to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Death','True Survivor'],
                         ['Pred Death','Pred Survivor'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./titanic_confusion_matrix.png")
        dataset_api.upload("./titanic_confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different state predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different state predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
       stub.deploy("titanic_prediction")
       with stub.run():
            f()

