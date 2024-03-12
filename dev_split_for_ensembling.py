import random
import pickle
from datasets import load_multitask_data
random.seed(11)

sst_dev = "data/ids-sst-train.csv"
para_dev = "data/quora-train.csv"
sts_dev = "data/sts-train.csv"

sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(
    sst_dev, para_dev, sts_dev, split ='train'
)

random.shuffle(sst_dev_data)
random.shuffle(para_dev_data)
random.shuffle(sts_dev_data)
sst_dev_train_data, para_dev_train_data, sts_dev_train_data = (
    [x for x in sst_dev_data[:int(len(sst_dev_data) * 0.5)]],
    [x for x in para_dev_data[:int(len(para_dev_data) * 0.5)]],
    [x for x in sts_dev_data[:int(len(sts_dev_data) * 0.5)]],
)

sst_dev_dev_data, para_dev_dev_data, sts_dev_dev_data = (
    [x for x in sst_dev_data[int(len(sst_dev_data) * 0.5):]],
    [x for x in para_dev_data[int(len(para_dev_data) * 0.5):]],
    [x for x in sts_dev_data[int(len(sts_dev_data) * 0.5):]],
)

with open('split_data_for_ensembling/dev_train_data.pkl', 'wb') as file:
    pickle.dump(
        (
            sst_dev_train_data, para_dev_train_data, sts_dev_train_data
        ),
        file,
    )
with open('split_data_for_ensembling/dev_dev_data.pkl', 'wb') as file:
    pickle.dump(
        (
            sst_dev_dev_data, para_dev_dev_data, sts_dev_dev_data
        ),
        file,
    )