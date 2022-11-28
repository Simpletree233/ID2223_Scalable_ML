import gradio as gr
import numpy as np
from PIL import Image
import requests
import pandas as pd
import hopsworks
import joblib
import os

#connect to hopsworks
project = hopsworks.login(project="test42",api_key_value=os.environ.get("HOPSWORKS_API_KEYS"))
fs = project.get_feature_store()

#get model
mr = project.get_model_registry()#connect to model registry
model = mr.get_model("titanic_model_modal", version=1) #retrieve model from hopsworks
model_dir = model.download() #download model to cur dir
model = joblib.load(model_dir + "/titanic_model.pkl") #load model from cur dir


def passenger(pclass,#index
              age,#float
              sibsp,#float
              parch,#float
              fare,#float
              sex,#index 0-male,1-female
              deck,# index abcdefgnt
              embarked# index cnqs
              ):
    deck_all="abcdefgnt"
    embarked_all="cnqs"
    deck_count=[0 for i in deck_all]
    deck_count[deck]=1
    embarked_count=[0 for i in embarked_all]
    embarked_count[embarked]=1
    
    input_df = pd.DataFrame({"pclass":[pclass+1],
              "age":[age],
              "sibsp":[sibsp],
              "parch":[parch],
              "fare":[fare],
              "sex_female":[sex],
              "sex_male":[1-sex],
              "deck_a":deck_count[0],
              "deck_b":deck_count[1],
              "deck_c":deck_count[2],
              "deck_d":deck_count[3],
              "deck_e":deck_count[4],
              "deck_f":deck_count[5],
              "deck_g":deck_count[6],
              "deck_n":deck_count[7],
              "deck_t":deck_count[8],
              "embarked_c":embarked_count[0],
              "embarked_n":embarked_count[1],
              "embarked_q":embarked_count[2],
              "embarked_s":embarked_count[3],
                  })
    # 'res' is a list of predictions returned as the label.
    #print(input_df)
    res = model.predict(input_df) #prediction from model based on input
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    if res==0:
        url="https://i.imgflip.com/5jvc2d.jpg"
        text="Dead 110 years ago rip"
    else:
        text="Dead but not on Titanic"
        url="https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcTv3XuQKvjiF_ZpUt8rKlsVBX--JXMpfa674H03N-aApj_HjK1S"
    img = Image.open(requests.get(url, stream=True).raw) #get image from github    
    
    return img,text

#create hugging face interface
demo = gr.Interface(
    passenger,
    [
        gr.Dropdown(["first", "second", "third"], type="index",label="Passenger Class"),
        gr.Slider(0, 80, value=25,label="Age"),
        gr.Slider(0, 10, step=1, value=0, label="Number of siblings/spouses aboard the Titanic"),
        gr.Slider(0, 10, step=1, value=0, label="Number of parents/children aboard the Titanic"),
        gr.Number(default=0, label="Passenger fare"),
        gr.Radio(["Male","Female"],type="index",label="Sex"),
        gr.Radio([f"Deck_{c}" for c in "ABCDEFGNT"],type="index",label="Deck (Select N if unknown)"),
        gr.Radio([f"Embarked_{e}" for e in "CNQS"],type="index",label="Embark point (Select N if unknown)")
        
    ],
    title="Titanic Survivor Predictive Analytics",
    description="Who could surive the titanic",
    allow_flagging="never",
    outputs=[gr.Image(type="pil"),gr.Label()]
    )

demo.launch()

