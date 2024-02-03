import json
import os
import librosa
import math

DATASET_PATH = "PaddedTest/"
JSON_PATH = "test_dataset.json"
TRACK_DURATION = 15

def save_chromagram(dataset_path,json_path,hop_length=512):
    data = {
        "mapping":[],
        "labels":[],
        "chromagram":[]
    }

    for i , (dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):

        if dirpath != dataset_path:
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: ".format(semantic_label))

            for f in filenames:
                file_path = os.path.join(dirpath,f)
                signal,sr = librosa.load(file_path)

                chromagram = librosa.feature.chroma_stft(y=signal,sr=sr,hop_length=512)
                data["chromagram"].append(chromagram.tolist())
                data["labels"].append(i-1)

    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)

save_chromagram(DATASET_PATH,JSON_PATH)