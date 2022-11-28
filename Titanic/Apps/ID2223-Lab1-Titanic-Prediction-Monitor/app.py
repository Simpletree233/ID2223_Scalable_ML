import gradio as gr
from PIL import Image
import hopsworks
import os
import imageio.v3 as iio
import inspect

project = hopsworks.login(project="test42",api_key_value=os.environ.get("HOPSWORKS_API_KEYS"))
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()
dataset_api.download("Resources/images/latest_passenger.png",overwrite=True)
dataset_api.download("Resources/images/actual_state.png",overwrite=True)
dataset_api.download("Resources/images/df_recent_state.png",overwrite=True)
dataset_api.download("Resources/images/titanic_confusion_matrix.png",overwrite=True)

def update():
    dataset_api.download("Resources/images/latest_passenger.png",overwrite=True)
    dataset_api.download("Resources/images/actual_state.png",overwrite=True)
    dataset_api.download("Resources/images/df_recent_state.png",overwrite=True)
    dataset_api.download("Resources/images/titanic_confusion_matrix.png",overwrite=True)

def update_latest_passenger_img():
    im = iio.imread('latest_passenger.png')
    return im

def update_actual_state_img():
    im = iio.imread('actual_state.png')
    return im

def update_df_recent_img():
    im = iio.imread('df_recent_state.png')
    return im

def update_confusion_matrix_img():
    im = iio.imread('titanic_confusion_matrix.png')
    return im

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          load=gr.Button("Load")
          load.click(fn=update)
      with gr.Column():
          refresh=gr.Button("Refresh")

    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Image")
          input_img = gr.Image("latest_passenger.png", elem_id="predicted-img")
          refresh.click(update_latest_passenger_img,outputs=input_img)
          
      with gr.Column():          
          gr.Label("Today's Actual Image")
          input_img = gr.Image("actual_state.png", elem_id="actual-img")
          refresh.click(update_actual_state_img,outputs=input_img)
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("df_recent_state.png", elem_id="recent-predictions")
          refresh.click(update_df_recent_img,outputs=input_img)
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("titanic_confusion_matrix.png", elem_id="confusion-matrix")
          refresh.click(update_confusion_matrix_img,outputs=input_img)




demo.launch()
