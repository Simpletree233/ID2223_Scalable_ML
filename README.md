# ID2223_Scalable_ML_Lab2

Here is the provided demo: [audio recognition](https://huggingface.co/spaces/Yuyang2022/Cantonese_speech_recognition)

Plus a video speech recognition: [Youtube video transcription](https://huggingface.co/spaces/WayneLinn/Cantonese_Speech_Recognition)

SPeech to speech translation: [From Cantonese to any chosen language](https://huggingface.co/spaces/Yuyang2022/Translation_yue_to_any)

## Model Performance
We used a model centric approach to improve the permance as possible under a constraint of computational resources. To achieve this, we modify the configurations as follows:

1. Initially, a Whisper-base model is trained and saved at checkpoints=500. The WER is around 80%.
1. Then, we keep training the model and save it at checkpint=1000, the WER doesnt change, it still remains at 80%.
1. For faster convergence, the learning rate is doubled and evaluation epoch is increased to accelerate training. 
1. the final result is still so-so, but we have trained another Whisper-small model. 
1. The difficulty is that the training data set for Chinese language is too large. In Huggingface, if we want to split the dataset we will have to write our own funciton.  

## Program refactor

1. Feature enginerring pipeline on CPU
> see `feature_pipeline.py`


2. Training pipeline on GPUs
> see `Model_training.ipynb`

3. Inference on Huggingface
> see Huggingface Space
