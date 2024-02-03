import numpy as np
import json

Dataset = "train_dataset.json"

def load_data(dataset_path):
    with open(dataset_path,"r") as fp:
        data = json.load(fp)

    x = np.array(data["chromagram"])

    return x

if __name__ == "__main__":
    x = load_data(Dataset)
    print(x.shape)