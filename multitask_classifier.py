'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceAllDataset,
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
    BatchSamplerAllDataset
)

from evaluation import model_eval_multitask, model_eval_test_multitask

from perturbation import SmartPerturbation

from torch.optim.lr_scheduler import LinearLR
from torch.optim import swa_utils

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

ALLNLI_FILENAME = 'data/AllNLI.jsonl'

def get_multi_negatives_ranking_loss(a, b, reduction='mean'):
    # Here, I'm following the implementation of MultiNegativesRankingLoss from the sentence_transformers library
    # for the case of cosine similarity specifically.
    # Reference: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
    embeddings_1_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    embeddings_2_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    similarity_matrix = torch.mm(a, b.transpose(0, 1))
    return F.cross_entropy(similarity_matrix, torch.arange(similarity_matrix.shape[0], device = a.device), reduction=reduction)

def get_unsupervised_simcse_loss(a, b):
    # In the paper, they use cosine similarity after tuning instead of a dot product (https://arxiv.org/pdf/2104.08821.pdf).
    # We use the dot product because it seems like it will do better out-of-the-box if we don't tune, since the range of
    # accuracies obtained at different amounts of tuning varies quite a bit.
    similarity_matrix = torch.mm(a, b.transpose(0, 1))
    return F.cross_entropy(similarity_matrix, torch.arange(similarity_matrix.shape[0], device = a.device), reduction='mean')

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        if config.load_model_state_dict_from_model_path is not None:
            self.bert = BertModel.from_pretrained(config.load_model_state_dict_from_model_path)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option in ('finetune', 'lp_ft', 'finetune_after_additional_pretraining'):
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.bert_embeddings_cache = {}
        self.config = config
        self.disable_complex_arch = config.disable_complex_arch
        if config.disable_complex_arch:
            print("Disabling complex arch")
            self.sentiment_linear = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
            self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.paraphrase_linear_for_dot = nn.Linear(config.hidden_size, config.paraphrase_embedding_size)
            self.paraphrase_linear_for_dot_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.paraphrase_final_linear = nn.Linear(config.hidden_size * 2 + config.paraphrase_embedding_size, 1)
            self.paraphrase_final_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.similarity_linear = nn.Linear(config.hidden_size, config.similarity_embedding_size)
            self.similarity_dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            # shared weights
            self.shared_linear_initial = nn.Linear(config.hidden_size, config.shared_linear_initial_size)
            self.shared_linear_initial_dropout = nn.Dropout(config.hidden_dropout_prob)
            # self.shared_linear_final = nn.Linear(config.shared_linear_initial_size, config.shared_linear_final_size)
            # self.shared_linear_final_dropout = nn.Dropout(config.hidden_dropout_prob)
            # dedicated weights for sentiment
            self.sentiment_linear = nn.Linear(config.hidden_size, config.sentiment_embedding_size)
            self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)
            # overarching weights for sentiment
            self.sentiment_overarch = nn.Linear(config.sentiment_embedding_size + config.shared_linear_final_size, N_SENTIMENT_CLASSES)
            # dedicated weights for paraphrase
            self.paraphrase_linear_for_dot = nn.Linear(config.hidden_size, config.paraphrase_embedding_size)
            self.paraphrase_final_linear = nn.Linear(config.hidden_size * 2 + config.paraphrase_embedding_size, config.paraphrase_embedding_size)
            self.paraphrase_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.paraphrase_final_dropout = nn.Dropout(config.hidden_dropout_prob)
            # overarching weights for paraphrase
            self.paraphrase_overarch = nn.Linear(config.paraphrase_embedding_size + config.shared_linear_final_size, 1)
            # dedicated weights for similarity
            self.similarity_linear = nn.Linear(config.hidden_size, config.similarity_embedding_size)
            self.similarity_dropout = nn.Dropout(config.hidden_dropout_prob)
            # overarching weights for similarity
            self.similarity_overarch = nn.Linear(config.similarity_embedding_size + config.shared_linear_final_size, config.similarity_embedding_size)
        if config.use_allnli_data:
            # TODO check on true shared arch implementation
            self.allnli_linear = nn.Linear(config.hidden_size, config.similarity_embedding_size)
            self.allnli_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, sent_ids, identifier):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        if self.config.option == 'pretrain' and sent_ids is not None:
            print("Should not enter caching flow now")
            for (i, sent_id) in enumerate(sent_ids):
                cache_key = (sent_id, identifier)
                if cache_key not in self.bert_embeddings_cache:
                    self.bert_embeddings_cache[cache_key] = self.bert(input_ids[i:i+1], attention_mask[i:i+1])['pooler_output']
            result = torch.stack([self.bert_embeddings_cache[(sent_id, identifier)] for sent_id in sent_ids]).squeeze(dim = 1)
            return result
        else:
            return self.bert(input_ids, attention_mask)['pooler_output']
    
    def get_shared_arch_output(self, bert_embedding):
        if self.disable_complex_arch:
            print("no shared arch when disabling complex arch")
            raise NotImplementedError
        # return self.shared_linear_final(
        # self.shared_linear_final_dropout(
        return self.shared_linear_initial(bert_embedding)

    def predict_sentiment_given_bert_embedding(self, bert_embedding):
        bert_embedding = self.sentiment_dropout(bert_embedding)
        if self.disable_complex_arch:
            return self.sentiment_linear(bert_embedding)
        shared_arch_output = self.get_shared_arch_output(bert_embedding)
        dedicated_arch_output = self.sentiment_linear(bert_embedding)
        return self.sentiment_overarch(
            torch.cat((shared_arch_output, dedicated_arch_output), dim=1)
        )

    def predict_sentiment_given_bert_input_embeds(self, input_embed, attention_mask):
        bert_embedding = self.bert.forward_given_input_embeds(input_embed, attention_mask)['pooler_output']
        return self.predict_sentiment_given_bert_embedding(bert_embedding)

    def predict_sentiment(self, input_ids, attention_mask, sent_ids=None):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        bert_embedding = self.forward(input_ids, attention_mask, sent_ids, 'sentiment')
        return self.predict_sentiment_given_bert_embedding(bert_embedding)

    # TODO address these functions when we re-enable complex arch
    def get_paraphrase_embedding_and_bert_embedding(self, input_id, attention_mask, sent_ids, identifier):
        if self.disable_complex_arch:
            bert_embedding = self.paraphrase_linear_for_dot_dropout(self.forward(input_id, attention_mask, sent_ids, identifier))
            embedding = self.paraphrase_linear_for_dot(bert_embedding)
            return embedding, bert_embedding
        else:
            raise NotImplementedError

    def predict_paraphrase_given_embeddings(self, embedding_1, embedding_2, bert_embedding_1, bert_embedding_2):
        combined_intermediate_output = torch.concat((bert_embedding_1, bert_embedding_2, embedding_1 * embedding_2), dim=1)
        return self.paraphrase_final_linear(
            self.paraphrase_final_dropout(combined_intermediate_output)
        ).view(-1)

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,
                           sent_ids=None):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO address this part when we re-try multiple negatives loss
        # embedding_1, bert_embedding_1 = self.get_paraphrase_embedding_and_bert_embedding(
        #     input_ids_1, attention_mask_1, sent_ids, 'para_1'
        # )
        # embedding_2, bert_embedding_2 = self.get_paraphrase_embedding_and_bert_embedding(
        #     input_ids_2, attention_mask_2, sent_ids, 'para_2'
        # )
        # return self.predict_paraphrase_given_embeddings(embedding_1, embedding_2, bert_embedding_1, bert_embedding_2)
        if self.disable_complex_arch:
            intermediate_output_1, bert_embedding_1 = self.get_paraphrase_embedding_and_bert_embedding(input_ids_1, attention_mask_1, sent_ids, 'para_1')
            intermediate_output_2, bert_embedding_2 = self.get_paraphrase_embedding_and_bert_embedding(input_ids_2, attention_mask_2, sent_ids, 'para_2')
            return self.predict_paraphrase_given_embeddings(intermediate_output_1, intermediate_output_2, bert_embedding_1, bert_embedding_2)
        bert_embedding_1 = self.paraphrase_dropout(self.forward(input_ids_1, attention_mask_1, sent_ids, 'para_1'))
        bert_embedding_2 = self.paraphrase_dropout(self.forward(input_ids_2, attention_mask_2, sent_ids, 'para_2'))
        shared_arch_output_1 = self.get_shared_arch_output(bert_embedding_1)
        shared_arch_output_2 = self.get_shared_arch_output(bert_embedding_2)
        dedicated_arch_output_1 = self.paraphrase_linear_for_dot(bert_embedding_1)
        dedicated_arch_output_2 = self.paraphrase_linear_for_dot(bert_embedding_2)
        dedicated_arch_intermediate = torch.concat((bert_embedding_1, bert_embedding_2, dedicated_arch_output_1 * dedicated_arch_output_2), dim=1)
        dedicated_arch_output = self.paraphrase_final_linear(self.paraphrase_final_dropout(dedicated_arch_intermediate))
        overarch_input = torch.cat(
            (
                shared_arch_output_1 * shared_arch_output_2,
                dedicated_arch_output
            ),
            dim=1
        )
        return self.paraphrase_overarch(overarch_input).view(-1)

    def get_similarity_embedding_given_bert_embedding(self, bert_embedding):
        bert_embedding = self.similarity_dropout(bert_embedding)
        if self.disable_complex_arch:
            return self.similarity_linear(bert_embedding)
        shared_arch_output = self.get_shared_arch_output(bert_embedding)
        dedicated_arch_output = self.similarity_linear(bert_embedding)
        return torch.cat((shared_arch_output, dedicated_arch_output), dim=1)
    
    def get_similarity_embedding(self, input_id, attention_mask, sent_ids, identifier):
        bert_embedding = self.forward(input_id, attention_mask, sent_ids, identifier)
        return self.get_similarity_embedding_given_bert_embedding(bert_embedding)

    def predict_similarity_given_embedding(self, embedding_1, embedding_2):
        return F.cosine_similarity(embedding_1, embedding_2) * 5.0

    def predict_similarity_given_bert_input_embeds(self, input_embed_1, attention_mask_1, input_embed_2, attention_mask_2):
        bert_embedding_1 = self.bert.forward_given_input_embeds(input_embed_1, attention_mask_1)['pooler_output']
        bert_embedding_2 = self.bert.forward_given_input_embeds(input_embed_2, attention_mask_2)['pooler_output']
        embedding_1 = self.get_similarity_embedding_given_bert_embedding(bert_embedding_1)
        embedding_2 = self.get_similarity_embedding_given_bert_embedding(bert_embedding_2)
        return self.predict_similarity_given_embedding(embedding_1, embedding_2)
    
    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,
                           sent_ids=None):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        embedding_1 = self.get_similarity_embedding(input_ids_1, attention_mask_1, sent_ids, 'similarity_1')
        embedding_2 = self.get_similarity_embedding(input_ids_2, attention_mask_2, sent_ids, 'similarity_2')
        return self.predict_similarity_given_embedding(embedding_1, embedding_2)

    def get_allnli_embedding(self, input_id, attention_mask, sent_ids, identifier):
        bert_embedding = self.forward(input_id, attention_mask, sent_ids, identifier)
        return self.allnli_linear(self.allnli_dropout(bert_embedding))

    def predict_allnli_given_embedding(self, embedding_1, embedding_2):
        return F.cosine_similarity(embedding_1, embedding_2)
    
    def predict_allnli(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, sent_ids=None):
        embedding_1 = self.get_allnli_embedding(input_ids_1, attention_mask_1, sent_ids, 'allnli_1')
        embedding_2 = self.get_allnli_embedding(input_ids_2, attention_mask_2, sent_ids, 'allnli_2')
        return self.predict_allnli_given_embedding(embedding_1, embedding_2)



def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def single_batch_train_sst(batch, model, optimizer, device, adv_teacher, debug=False):
        b_ids, b_mask, b_labels, b_sent_ids = (batch['token_ids'],
                                    batch['attention_mask'], batch['labels'], batch['sent_ids'])
        if debug:
            print(b_sent_ids[:5], b_ids[:5], b_sent_ids[:5], b_labels[:5])
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        optimizer.zero_grad()
        logits = model.predict_sentiment(b_ids, b_mask, b_sent_ids)
        if debug:
            print("sst", logits[:5, :], b_labels[:5])
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        return train_loss

def single_epoch_train_sst(sst_train_dataloader, epoch, model, optimizer, device, adv_teacher, debug=False):
    train_loss = 0
    num_batches = 0
    for batch in tqdm(sst_train_dataloader, desc=f'train-sst-{epoch}', disable=TQDM_DISABLE):
        train_loss += single_batch_train_sst(batch, model, optimizer, device, adv_teacher, debug)
        num_batches += 1
        if debug and num_batches >= 5:
            break
    train_loss = train_loss / num_batches
    return train_loss

def single_batch_train_para(batch, model, optimizer, device, adv_teacher, grad_scaling_factor_for_para, debug=False):
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels, b_sent_ids = (
        batch['token_ids_1'],
        batch['attention_mask_1'],
        batch['token_ids_2'],
        batch['attention_mask_2'],
        batch['labels'],
        batch['sent_ids'],
    )
    if debug:
        print(b_sent_ids[:5], b_labels[:5])
    b_ids_1 = b_ids_1.to(device)
    b_mask_1 = b_mask_1.to(device)
    b_ids_2 = b_ids_2.to(device)
    b_mask_2 = b_mask_2.to(device)
    b_labels = b_labels.to(device)

    optimizer.zero_grad()
    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_sent_ids)
    # embeddings_1, bert_embeddings_1 = model.get_paraphrase_embedding_and_bert_embedding(b_ids_1, b_mask_1, b_sent_ids, 'para_1')
    # embeddings_2, bert_embeddings_2 = model.get_paraphrase_embedding_and_bert_embedding(b_ids_2, b_mask_2, b_sent_ids, 'para_2')
    # logits = model.predict_paraphrase_given_embeddings(embeddings_1, embeddings_2, bert_embeddings_1, bert_embeddings_2)
    multi_negatives_ranking_loss = 0 # get_multi_negatives_ranking_loss(embeddings_1, embeddings_2, reduction = 'none')
    # multi_negatives_ranking_loss = torch.sum((b_labels == 1) * multi_negatives_ranking_loss) / torch.sum(b_labels == 1)
    loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1).float(), reduction='sum') / args.batch_size
    if debug:
        print("para", logits[:5], b_labels[:5])
        print("para loss", loss, "multi negatives ranking loss", multi_negatives_ranking_loss)
    loss = grad_scaling_factor_for_para * (loss + 0.5 * multi_negatives_ranking_loss)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    return train_loss


def single_epoch_train_para(para_train_dataloader, epoch, model, optimizer, device, adv_teacher, grad_scaling_factor_for_para, debug=False):
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-para-{epoch}', disable=TQDM_DISABLE):
        train_loss += single_batch_train_para(batch, model, optimizer, device, adv_teacher, grad_scaling_factor_for_para, debug)
        num_batches += 1
        if debug and num_batches >= 5:
            break
    train_loss = train_loss / num_batches
    return train_loss

def single_batch_train_sts(batch, model, optimizer, device, adv_teacher, enable_unsupervised_simcse, debug=False):
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels, b_sent_ids = (
        batch['token_ids_1'],
        batch['attention_mask_1'],
        batch['token_ids_2'],
        batch['attention_mask_2'],
        batch['labels'],
        batch['sent_ids'],
    )
    if debug:
        print(b_sent_ids[:5], b_labels[:5])
    b_ids_1 = b_ids_1.to(device)
    b_mask_1 = b_mask_1.to(device)
    b_ids_2 = b_ids_2.to(device)
    b_mask_2 = b_mask_2.to(device)
    b_labels = b_labels.to(device)

    optimizer.zero_grad()
    # predictions = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_sent_ids)
    embeddings_1 = model.get_similarity_embedding(b_ids_1, b_mask_1, b_sent_ids, 'similarity_1')
    embeddings_2 = model.get_similarity_embedding(b_ids_2, b_mask_2, b_sent_ids, 'similarity_2')
    if enable_unsupervised_simcse:
        embeddings_1_copy_with_diff_dropout = model.get_similarity_embedding(b_ids_1, b_mask_1, b_sent_ids, 'similarity_1')
        embeddings_2_copy_with_diff_dropout = model.get_similarity_embedding(b_ids_2, b_mask_2, b_sent_ids, 'similarity_2')
        unsupervised_simcse_mask = torch.randint(0, 2, (embeddings_1.shape[0], 1)).bool().to(device)
        embeddings_for_unsupervised_simcse = torch.where(unsupervised_simcse_mask, embeddings_1, embeddings_2)
        embeddings_for_unsupervised_simcse_copy = torch.where(unsupervised_simcse_mask, embeddings_1_copy_with_diff_dropout, embeddings_2_copy_with_diff_dropout)
        if debug:
            assert(not (torch.all(embeddings_for_unsupervised_simcse == embeddings_for_unsupervised_simcse_copy)))
        unsupervised_simcse_loss = get_unsupervised_simcse_loss(embeddings_for_unsupervised_simcse, embeddings_for_unsupervised_simcse_copy)

    predictions = model.predict_similarity_given_embedding(embeddings_1, embeddings_2)
    multi_negatives_ranking_loss = get_multi_negatives_ranking_loss(embeddings_1, embeddings_2, reduction = 'none')
    # We should weight the multi-negatives-ranking-loss by the similarity of the texts.
    # Try soft-weighting based on the similarity of the texts.
    C = sum(np.exp(np.arange(1, 5))) # TODO fix bug here
    multi_negatives_ranking_loss = torch.sum((b_labels >= 1) * torch.exp(b_labels) / C * multi_negatives_ranking_loss) / torch.sum((b_labels >= 1))
    loss = F.mse_loss(predictions, b_labels.view(-1).float(), reduction='sum') / args.batch_size
    adv_loss = 0
    if adv_teacher is not None:
        adv_loss = adv_teacher.forward(model, predictions, b_ids_1, b_mask_1, b_ids_2, b_mask_2, 'similarity')
    if debug:
        print("sts", predictions[:5], b_labels[:5], loss, multi_negatives_ranking_loss)
    loss = loss + 10 * multi_negatives_ranking_loss + 50 * adv_loss
    if enable_unsupervised_simcse:
        loss = loss + 10 * unsupervised_simcse_loss
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    return train_loss

def single_batch_train_allnli(batch, model, optimizer, device, debug=False):
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels, b_sent_ids = (
        batch['token_ids_1'],
        batch['attention_mask_1'],
        batch['token_ids_2'],
        batch['attention_mask_2'],
        batch['labels'],
        batch['sent_ids'],
    )
    if debug:
        print(b_sent_ids[:5], b_labels[:5])
    b_ids_1 = b_ids_1.to(device)
    b_mask_1 = b_mask_1.to(device)
    b_ids_2 = b_ids_2.to(device)
    b_mask_2 = b_mask_2.to(device)
    b_labels = b_labels.to(device)

    optimizer.zero_grad()
    embeddings_1 = model.get_allnli_embedding(b_ids_1, b_mask_1, b_sent_ids, 'allnli_1')
    embeddings_2 = model.get_allnli_embedding(b_ids_2, b_mask_2, b_sent_ids, 'allnli_2')
    loss = get_multi_negatives_ranking_loss(embeddings_1, embeddings_2, reduction = 'mean')
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    return train_loss

def single_epoch_train_sts(sts_train_dataloader, epoch, model, optimizer, device, adv_teacher, enable_unsupervised_simcse, debug=False):
    train_loss = 0
    num_batches = 0
    for batch in tqdm(sts_train_dataloader, desc=f'train-sts-{epoch}', disable=TQDM_DISABLE):
        train_loss += single_batch_train_sts(batch, model, optimizer, device, adv_teacher, enable_unsupervised_simcse, debug)
        num_batches += 1
        if debug and num_batches >= 5:
            break
    train_loss = train_loss / num_batches
    return train_loss

def single_epoch_train_all(
        train_dataloader,
        epoch,
        model,
        optimizer,
        device,
        adv_teachers,
        enable_unsupervised_simcse,
        use_allnli_data,
        grad_scaling_factor_for_para,
        debug=False,
        exclude_sst = False,
        exclude_para = False,
        exclude_sts = False
    ):
    adv_teacher_similarity, adv_teacher_paraphrase, adv_teacher_sentiment = adv_teachers
    sst_train_loss, num_sst_batches = 0, 0
    para_train_loss, num_para_batches = 0, 0
    sts_train_loss, num_sts_batches = 0, 0
    if use_allnli_data:
        allnli_train_loss, num_allnli_batches = 0, 0
    for batch in tqdm(train_dataloader, desc=f'train-all-{epoch}', disable=TQDM_DISABLE):
        if batch['dataset_name'] == 'sentiment' and not exclude_sst:
            sst_train_loss += single_batch_train_sst(batch, model, optimizer, device, adv_teacher_sentiment, debug)
            num_sst_batches += 1
        elif batch['dataset_name'] == 'paraphrase' and not exclude_para:
            para_train_loss += single_batch_train_para(batch, model, optimizer, device, adv_teacher_paraphrase, grad_scaling_factor_for_para, debug)
            num_para_batches += 1
        elif batch['dataset_name'] == 'similarity' and not exclude_sts:
            sts_train_loss += single_batch_train_sts(batch, model, optimizer, device, adv_teacher_similarity, enable_unsupervised_simcse, debug)
            num_sts_batches += 1
        elif batch['dataset_name'] == 'allnli' and use_allnli_data:
            allnli_train_loss += single_batch_train_allnli(batch, model, optimizer, device, debug)
            num_allnli_batches += 1
        if debug and num_sst_batches + num_para_batches + num_sts_batches >= 5:
            break
    def get_loss(loss, num_batches, desc):
        print(f"Loss of {loss} for {desc} after {num_batches} batches")
        if num_batches == 0:
            return -1
        return loss / num_batches
    sst_train_loss = get_loss(sst_train_loss, num_sst_batches, 'sst')
    para_train_loss = get_loss(para_train_loss, num_para_batches, 'para')
    sts_train_loss = get_loss(sts_train_loss, num_sts_batches, 'sts')
    if use_allnli_data:
        allnli_train_loss = get_loss(allnli_train_loss, num_allnli_batches, 'allnli')
        return sst_train_loss, para_train_loss, sts_train_loss, allnli_train_loss
    else:
        return sst_train_loss, para_train_loss, sts_train_loss

def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    device = torch.device('cpu')
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.use_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
    # Create the data and its corresponding datasets and dataloader.
    if args.use_allnli_data:
        sst_train_data, num_labels,para_train_data, sts_train_data, allnli_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train', allnli_filename=ALLNLI_FILENAME, allnli_split='train')
        sst_dev_data, num_labels,para_dev_data, sts_dev_data, allnli_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train', allnli_filename=ALLNLI_FILENAME, allnli_split='dev')
    else:
        sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
        sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')
    if not args.use_even_batching:
        assert(not args.use_allnli_data)
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sst_train_data.collate_fn)
        para_train_data = SentencePairDataset(para_train_data, args)
        para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
        sts_train_data = SentencePairDataset(sts_train_data, args, isRegression = True)

        sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    else:
        all_train_datasets = [sst_train_data, para_train_data, sts_train_data]
        all_dev_datasets = [sst_dev_data, para_dev_data, sts_dev_data]
        if args.use_allnli_data:
            all_train_datasets.append(allnli_train_data)
            all_dev_datasets.append(allnli_dev_data)
        train_data = SentenceAllDataset(all_train_datasets, args)
        train_batch_sampler = BatchSamplerAllDataset(train_data.datasets, args.batch_size, shuffle = True)
        dev_data = SentenceAllDataset(all_dev_datasets, args)
        dev_batch_sampler = BatchSamplerAllDataset(dev_data.datasets, args.batch_size, shuffle = False)
        train_dataloader = DataLoader(train_data, collate_fn = train_data.collate_fn, batch_sampler = train_batch_sampler)
        dev_dataloader = DataLoader(dev_data, collate_fn = dev_data.collate_fn, batch_sampler = dev_batch_sampler)
        # test the dataloader
        '''
        for epoch in range(args.epochs):
            print("Epoch", epoch)
            batches_by_dataset = {}
            for batch in tqdm(train_dataloader, desc=f'testing-new-dataloader', disable=TQDM_DISABLE):
                dataset_name = batch['dataset_name']
                batches_by_dataset[dataset_name] = batches_by_dataset.get(dataset_name, 0) + 1
            print("Batches by dataset is", batches_by_dataset)
        '''
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                collate_fn=sst_dev_data.collate_fn)
        para_dev_data = SentencePairDataset(para_dev_data, args)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                collate_fn=para_dev_data.collate_fn)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression = True)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)
    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'sentiment_embedding_size': 128,
              'similarity_embedding_size': 128,
              'paraphrase_embedding_size': 128,
              'shared_linear_initial_size': 128,
              # currently we don't use final - must be same dim as initial
              'shared_linear_final_size': 128,
              'hidden_size': 768,
              'data_dir': '.',
              'load_model_state_dict_from_model_path': args.load_model_state_dict_from_model_path if args.option == 'finetune_after_additional_pretraining' else None,
              'disable_complex_arch': args.disable_complex_arch,
              'use_allnli_data': args.use_allnli_data,
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    if args.option == 'lp_ft' and args.load_model_state_dict_from_model_path is not None:
        print(f"Loading model from {args.load_model_state_dict_from_model_path}")
        saved = torch.load(args.load_model_state_dict_from_model_path)
        model.load_state_dict(saved['model'])
        print("Loaded model state")
        print("Old config was", saved['model_config'])
        print("New config is", model.config)
    model = model.to(device)
    
    lr = args.lr
    print("Learning rate of", lr, "for", args.epochs, "epochs")
    optimizer = AdamW(model.parameters(), weight_decay=args.weight_decay, lr=lr)
    if args.linear_lr_decay_with_swa:
        swa_model = swa_utils.AveragedModel(model)
        swa_start = args.epochs * 0.75
        scheduler = LinearLR(optimizer, start_factor = 1.0, end_factor = 0.2, total_iters = swa_start)
        swa_scheduler = swa_utils.SWALR(optimizer, swa_lr = 5 * lr, anneal_epochs = int(0.1 * args.epochs))

    best_dev_acc = 0
    adv_teacher_similarity = None
    adv_teacher_paraphrase = None
    adv_teacher_sentiment = None
    if args.adv_train:
        adv_teacher_similarity = SmartPerturbation()
        adv_teacher_paraphrase = SmartPerturbation()
        adv_teacher_sentiment = SmartPerturbation()

    # Run for the specified number of epochs.
    exclude_sts = False
    exclude_para = False
    exclude_sst = False
    debug = False
    for epoch in range(args.epochs):
        model.train()
        if not args.use_even_batching:
            assert(not args.use_allnli_data)
            if exclude_sts:
                sts_train_loss = -1
            else:
                sts_train_loss = single_epoch_train_sts(sts_train_dataloader, epoch, model, optimizer, device, adv_teacher_similarity, args.enable_unsupervised_simcse, debug = debug)
            if exclude_para:
                para_train_loss = -1
            else:
                para_train_loss = single_epoch_train_para(para_train_dataloader, epoch, model, optimizer, device, adv_teacher_paraphrase, args.grad_scaling_factor_for_para, debug = debug)
            if exclude_sst:
                sst_train_loss = -1
            else:
                sst_train_loss = single_epoch_train_sst(sst_train_dataloader, epoch, model, optimizer, device, adv_teacher_sentiment, debug = debug)
        else:
            losses = single_epoch_train_all(
                train_dataloader,
                epoch,
                model,
                optimizer,
                device,
                (adv_teacher_similarity, adv_teacher_paraphrase, adv_teacher_sentiment),
                args.enable_unsupervised_simcse,
                args.use_allnli_data,
                args.grad_scaling_factor_for_para,
                debug = debug,
                exclude_sst = exclude_sst,
                exclude_para = exclude_para,
                exclude_sts = exclude_sts
            )
            if not args.use_allnli_data:
                sst_train_loss, para_train_loss, sts_train_loss = losses
            else:
                sst_train_loss, para_train_loss, sts_train_loss, allnli_train_loss = losses
                print(sst_train_loss, para_train_loss, sts_train_loss, allnli_train_loss)
        print(f"Epoch {epoch}: train loss :: {sst_train_loss :.3f}, {para_train_loss :.3f}, {sts_train_loss :.3f}")
        if debug:
            print("Learning rates are:", )
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
        if args.linear_lr_decay_with_swa:
            if epoch > swa_start:
                if debug:
                    print("swa step")
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                if debug:
                    print("scheduler step")
                scheduler.step()
        '''
        print(f"Epoch {epoch}: train data stats")
        sst_train_acc, _, _, para_train_acc, _, _, sts_train_acc, _, _ = model_eval_multitask(
            sst_train_dataloader,
            para_train_dataloader,
            sts_train_dataloader,
            model,
            device,
            limit_batches = 100,
            exclude_sts = exclude_sts,
            exclude_para = exclude_para,
            exclude_sst = exclude_sst,
        )
        '''
        print(f"Epoch {epoch}: dev data stats")
        sst_dev_acc, _, _, para_dev_acc, _, _, sts_dev_acc, _, _ = model_eval_multitask(
            sst_dev_dataloader,
            para_dev_dataloader,
            sts_dev_dataloader,
            model if ((not args.linear_lr_decay_with_swa) or epoch <= swa_start) else swa_model.module,
            device,
            limit_batches = None if not debug else 50,
            exclude_sts = exclude_sts,
            exclude_para = exclude_para,
            exclude_sst = exclude_sst,
        )
        sts_dev_acc = (1.0 + sts_dev_acc) / 2.0 # normalize Pearson correlation to be in [0, 1] range
        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        if exclude_para:
            dev_acc = (sst_dev_acc + sts_dev_acc) / 2
        else:
            dev_acc = (sst_dev_acc + para_dev_acc + sts_dev_acc) / 3
        if dev_acc > best_dev_acc:
            print(f"Dev acc {dev_acc} is better than best dev acc {best_dev_acc}")
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
        else:
            print(f"Dev acc {dev_acc} is not better than best dev acc {best_dev_acc}")
        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        device = torch.device('cpu')
        if args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        elif args.use_gpu and torch.backends.mps.is_available():
            device = torch.device('mps')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune', 'lp_ft', 'finetune_after_additional_pretraining'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--weight_decay", type=float, help="weight decay", default=0.0)
    parser.add_argument(
        "--load_model_state_dict_from_model_path",
        type=str,
        help='Only loads model state dict; does NOT load optimizer state, config, etc.; only for LP+FT'
    )
    parser.add_argument("--use_even_batching", action='store_true')
    parser.add_argument("--adv_train", action='store_true')
    parser.add_argument("--disable_complex_arch", action='store_true')
    parser.add_argument("--use_allnli_data", action='store_true')
    parser.add_argument("--enable_unsupervised_simcse", action='store_true')
    parser.add_argument("--grad_scaling_factor_for_para", type=float, default=1.0)
    parser.add_argument("--linear_lr_decay_with_swa", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    assert(args.load_model_state_dict_from_model_path is None or args.option in ('lp_ft', 'finetune_after_additional_pretraining'))
    if args.option == 'finetune_after_additional_pretraining':
        assert(args.load_model_state_dict_from_model_path is not None)
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-{args.batch_size}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
