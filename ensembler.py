import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
import pickle

from datasets import (
    SentenceAllDataset,
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
    BatchSamplerAllDataset
)

from evaluation import model_eval_for_distillation
from multitask_classifier import MultitaskBERT

def get_sst_acc(sst_sent_ids_to_predictions, sst_sent_ids_to_labels):
    sst_sent_ids = list(sst_sent_ids_to_predictions.keys())
    sst_predictions = [torch.argmax(sst_sent_ids_to_predictions[x][-1]).cpu().numpy() for x in sst_sent_ids]
    sst_labels = [sst_sent_ids_to_labels[x] for x in sst_sent_ids]
    print("Sentiment accuracy is", np.mean(np.array(sst_predictions) == np.array(sst_labels)))

def get_para_acc(para_sent_ids_to_predictions, para_sent_ids_to_labels):
    para_sent_ids = list(para_sent_ids_to_predictions.keys())
    para_predictions = [para_sent_ids_to_predictions[x][-1].round().cpu().numpy() for x in para_sent_ids]
    para_labels = [para_sent_ids_to_labels[x] for x in para_sent_ids]
    print("Paraphrase accuracy is", np.mean(np.array(para_predictions) == np.array(para_labels)))

def get_sts_pearson(sts_sent_ids_to_predictions, sts_sent_ids_to_labels):
    sts_sent_ids = list(sts_sent_ids_to_predictions.keys())
    sts_predictions = [sts_sent_ids_to_predictions[x][-1].cpu().numpy() for x in sts_sent_ids]
    sts_labels = [sts_sent_ids_to_labels[x] for x in sts_sent_ids]
    pearson_mat = np.corrcoef(sts_predictions, sts_labels)
    sts_corr = pearson_mat[1][0]
    print("STS pearson is", sts_corr)

model_paths = [
    'best_as_of_mar_10_morning.pt',
    'para_0_1_num_embeddings_3_mar_11_evening.pt',
    'para_0_3_model.pt',
    'para_distillation_mar_10.pt'
]

sst_dev = "data/ids-sst-dev.csv"
para_dev = "data/quora-dev.csv"
sts_dev = "data/sts-dev.csv"

device = torch.device('cuda')
batch_size = 16

# Create the data and its corresponding datasets and dataloader.
sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(
    sst_dev, para_dev, sts_dev, split ='train'
)
sst_dev_data = SentenceClassificationDataset(sst_dev_data, None)
sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=True, batch_size=16,
                                collate_fn=sst_dev_data.collate_fn)
para_dev_data = SentencePairDataset(para_dev_data, None)
para_dev_dataloader = DataLoader(para_dev_data, shuffle=True, batch_size=16,
                                collate_fn=para_dev_data.collate_fn)
sts_dev_data = SentencePairDataset(sts_dev_data, None, isRegression = True)

sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=True, batch_size=16,
                                collate_fn=para_dev_data.collate_fn)

sst_sent_ids_to_predictions, para_sent_ids_to_predictions, sts_sent_ids_to_predictions = {}, {}, {}
sst_sent_ids_to_labels, para_sent_ids_to_labels, sts_sent_ids_to_labels = {}, {}, {}
for path in model_paths:
    # Init model.
    saved = torch.load(path)
    saved['model_config'].add_distillation_from_predictions_path = False
    model = MultitaskBERT(saved['model_config'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print("Loaded from path:", path)
    distillation_eval = model_eval_for_distillation(
        sst_dev_dataloader,
        para_dev_dataloader,
        sts_dev_dataloader,
        model,
        device,
        limit_batches=None,
        include_labels=True,
    )
    (
        sst_y_logits, sst_sent_ids, sst_labels,
        para_y_logits, para_sent_ids, para_labels,
        sts_y_logits, sts_sent_ids, sts_labels,
    ) = distillation_eval
    for (i, x) in enumerate(sst_sent_ids):
        if x not in sst_sent_ids_to_predictions:
            sst_sent_ids_to_predictions[x] = []
        sst_sent_ids_to_predictions[x].append(F.softmax(torch.tensor(sst_y_logits[i]), 0))
        if x in sst_sent_ids_to_labels:
            assert sst_sent_ids_to_labels[x] == sst_labels[i]
        else:
            sst_sent_ids_to_labels[x] = sst_labels[i]
    for (i, x) in enumerate(para_sent_ids):
        if x not in para_sent_ids_to_predictions:
            para_sent_ids_to_predictions[x] = []
        para_sent_ids_to_predictions[x].append((torch.tensor(para_y_logits[i])).sigmoid())
        if x in para_sent_ids_to_labels:
            assert para_sent_ids_to_labels[x] == para_labels[i]
        else:
            para_sent_ids_to_labels[x] = para_labels[i]
    for (i, x) in enumerate(sts_sent_ids):
        if x not in sts_sent_ids_to_predictions:
            sts_sent_ids_to_predictions[x] = []
        sts_sent_ids_to_predictions[x].append(torch.tensor(sts_y_logits[i]))
        if x in sts_sent_ids_to_labels:
            assert sts_sent_ids_to_labels[x] == sts_labels[i]
        else:
            sts_sent_ids_to_labels[x] = sts_labels[i]
    print("Got predictions")
    get_sst_acc(sst_sent_ids_to_predictions, sst_sent_ids_to_labels)
    get_para_acc(para_sent_ids_to_predictions, para_sent_ids_to_labels)
    get_sts_pearson(sts_sent_ids_to_predictions, sts_sent_ids_to_labels)

# Average the predictions.
for k in sst_sent_ids_to_predictions:
    sst_sent_ids_to_predictions[k] = [torch.stack(sst_sent_ids_to_predictions[k]).mean(dim=0)]
for k in para_sent_ids_to_predictions:
    para_sent_ids_to_predictions[k] = [torch.stack(para_sent_ids_to_predictions[k]).mean(dim=0)]
for k in sts_sent_ids_to_predictions:
    sts_sent_ids_to_predictions[k] = [torch.stack(sts_sent_ids_to_predictions[k]).mean(dim=0)]

print("Averaged predictions, final eval")
get_sst_acc(sst_sent_ids_to_predictions, sst_sent_ids_to_labels)
get_para_acc(para_sent_ids_to_predictions, para_sent_ids_to_labels)
get_sts_pearson(sts_sent_ids_to_predictions, sts_sent_ids_to_labels)
