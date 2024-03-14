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

from evaluation import model_test_eval_for_distillation
from multitask_classifier import MultitaskBERT

def save_sst(sst_sent_ids_to_predictions):
    sst_sent_ids = list(sst_sent_ids_to_predictions.keys())
    sst_predictions = [torch.argmax(sst_sent_ids_to_predictions[x][-1]).cpu().numpy() for x in sst_sent_ids]
    with open(save_sst_test, "w+") as f:
        f.write(f"id \t Predicted_Sentiment \n")
        for p, s in zip(sst_sent_ids, sst_predictions):
            f.write(f"{p} , {s} \n")
        print("Saved sst")

def save_para(para_sent_ids_to_predictions):
    para_sent_ids = list(para_sent_ids_to_predictions.keys())
    para_predictions = [para_sent_ids_to_predictions[x][-1].round().cpu().numpy() for x in para_sent_ids]
    with open(save_para_test, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(para_sent_ids, para_predictions):
            f.write(f"{p} , {s} \n")
    print("Saved para")

def save_sts(sts_sent_ids_to_predictions):
    sts_sent_ids = list(sts_sent_ids_to_predictions.keys())
    sts_predictions = [sts_sent_ids_to_predictions[x][-1].cpu().numpy() for x in sts_sent_ids]
    with open(save_sts_test, "w+") as f:
        f.write(f"id \t Predicted_Similiary \n")
        for p, s in zip(sts_sent_ids, sts_predictions):
            f.write(f"{p} , {s} \n")
    print("Saved sts")

model_paths = [
    'best_as_of_mar_10_morning.pt',
    'para_0_1_num_embeddings_3_mar_11_evening.pt',
    'para_0_3_model.pt',
    'para_distillation_mar_10.pt',
    'shared_allnli_weights_mar_12.pt',
    'distilled_model_mar_13.pt',
]
sst_test = "data/ids-sst-test-student.csv"
para_test = "data/quora-test-student.csv"
sts_test = "data/sts-test-student.csv"
save_sst_test = "predictions/sst-test-output.csv"
save_para_test = "predictions/para-test-output.csv"
save_sts_test = "predictions/sts-test-output.csv"

device = torch.device('cuda')
batch_size = 16

# Create the data and its corresponding datasets and dataloader.
sst_test_data, _, para_test_data, sts_test_data = load_multitask_data(
    sst_test, para_test, sts_test, split ='test'
)

sst_test_data = SentenceClassificationTestDataset(sst_test_data, None)
sst_test_dataloader = DataLoader(
    sst_test_data, shuffle=True, batch_size=16,
    collate_fn=sst_test_data.collate_fn
)
para_test_data = SentencePairTestDataset(para_test_data, None)
para_test_dataloader = DataLoader(
    para_test_data, shuffle=True, batch_size=16,
    collate_fn=para_test_data.collate_fn
)
sts_test_data = SentencePairTestDataset(sts_test_data, None)
sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=16,
                                    collate_fn=sts_test_data.collate_fn)

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
    distillation_eval = model_test_eval_for_distillation(
        sst_test_dataloader,
        para_test_dataloader,
        sts_test_dataloader,
        model,
        device,
    )
    (
        sst_y_logits, sst_sent_ids,
        para_y_logits, para_sent_ids,
        sts_y_logits, sts_sent_ids,
    ) = distillation_eval
    for (i, x) in enumerate(sst_sent_ids):
        if x not in sst_sent_ids_to_predictions:
            sst_sent_ids_to_predictions[x] = []
        sst_sent_ids_to_predictions[x].append(F.softmax(torch.tensor(sst_y_logits[i]), 0))
    for (i, x) in enumerate(para_sent_ids):
        if x not in para_sent_ids_to_predictions:
            para_sent_ids_to_predictions[x] = []
        para_sent_ids_to_predictions[x].append((torch.tensor(para_y_logits[i])).sigmoid())
    for (i, x) in enumerate(sts_sent_ids):
        if x not in sts_sent_ids_to_predictions:
            sts_sent_ids_to_predictions[x] = []
        sts_sent_ids_to_predictions[x].append(torch.tensor(sts_y_logits[i]))
    print("Got predictions")

# Average the predictions.
printed = False
for k in sst_sent_ids_to_predictions:
    # Obtained from softmax regression on dev dev data
    # sst_weights = torch.tensor([0.3538, 0.4908, 0.6168, -0.2704, 0.7000, 0.3866])
    sst_weights = torch.tensor([1.0 / 6 for i in range(6)])
    current_predictions = torch.stack(sst_sent_ids_to_predictions[k])
    if not printed:
        print(sst_weights.shape, current_predictions.shape)
        printed = True
    sst_sent_ids_to_predictions[k] = [(current_predictions * sst_weights.unsqueeze(dim = 1)).mean(dim=0)]
for k in para_sent_ids_to_predictions:
    # Obtained from logistic regression on dev dev data
    # para_weights = torch.tensor([0.0145, 0.1204, 0.0763, -0.0173, 0.2543, 0.0810])
    para_weights = torch.tensor([1.0 / 6 for i in range(6)])
    current_predictions = torch.stack(para_sent_ids_to_predictions[k])
    if not printed:
        print(para_weights.shape, current_predictions.shape)
        printed = True
    current_logits = torch.log(current_predictions) - torch.log(1 - current_predictions)
    para_sent_ids_to_predictions[k] = [(para_weights * current_logits).sum().sigmoid()]
for k in sts_sent_ids_to_predictions:
    sts_sent_ids_to_predictions[k] = [torch.stack(sts_sent_ids_to_predictions[k]).mean(dim=0)]
print("Averaged predictions, save them")
save_sst(sst_sent_ids_to_predictions)
save_para(para_sent_ids_to_predictions)
save_sts(sts_sent_ids_to_predictions)
