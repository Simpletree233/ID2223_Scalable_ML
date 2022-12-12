import modal
import hopsworks
import os

LOCAL=False

if LOCAL==False:
    stub=modal.Stub()
    image=modal.Image.debian_slim().apt_install(["ffmpeg","git"]).pip_install(["ffmpeg",
                                                                               "hopsworks",
                                                                               "datasets",
                                                                               "librosa",
                                                                               "evaluate",
                                                                               "jiwer",
                                                                               "torch",
                                                                               "torchaudio"
                                                                               ])
    

    @stub.function(image=image, timeout=1*60*60, schedule=modal.Period(days=1), secret=modal.Secret.from_name("lab1"))
    def f():
        g()

def g():
    import hopsworks
    from huggingface_hub import login
    os.system("python -m pip install git+https://github.com/huggingface/transformers")
    
    from datasets import load_dataset, DatasetDict
    login(token=os.environ["HUGGINGFACE_API_KEY"])

    common_voice = DatasetDict()

    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "yue", split="train", use_auth_token=True)
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "yue", split="test", use_auth_token=True)
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="zh", task="transcribe")
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="zh", task="transcribe")
    from datasets import Audio

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)
    common_voice.save_to_disk("common_voice")

    hopsworks_path="/Projects/test42/Lab2/common_voice"
    train_path="/Projects/test42/Lab2/common_voice/train"
    test_path="/Projects/test42/Lab2/common_voice/test"
    project = hopsworks.login(project="test42")
    dataset_api = project.get_dataset_api()

    uploaded_file_path = dataset_api.upload(
        local_path = "./common_voice/dataset_dict.json", 
        upload_path = hopsworks_path, overwrite=True)

    uploaded_file_path = dataset_api.upload(
        local_path = "./common_voice/train/state.json", 
        upload_path = train_path, overwrite=True)

    uploaded_file_path = dataset_api.upload(
        local_path = "./common_voice/test/state.json", 
        upload_path = test_path, overwrite=True)

    uploaded_file_path = dataset_api.upload(
        local_path = "./common_voice/train/dataset.arrow", 
        upload_path = train_path, overwrite=True)

    uploaded_file_path = dataset_api.upload(
        local_path = "./common_voice/test/dataset.arrow", 
        upload_path = test_path, overwrite=True)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
