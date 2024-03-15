# This file is taken from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/perturbation.py
# in the MT-DNN repository (associated with the SMART paper: https://arxiv.org/abs/1911.03437)
# and is originally authored by the SMART authors.
# We adapt it to our use case here in order to use the smoothness-inducing regularization
# technique.
import torch
import logging
import torch.nn.functional as F

def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
    ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
    if reduce:
        return (p * (rp - ry) * 2).sum() / bs
    else:
        return (p * (rp - ry) * 2).sum()

def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise


class SmartPerturbation:
    def __init__(
        self,
        epsilon=1e-6,
        # multi_gpu_on=False,
        step_size=1e-3,
        noise_var=1e-5,
        norm_p="inf",
        k=1,
        # fp16=False,
        # encoder_type=EncoderModelType.BERT,
        # loss_map=[],
        # norm_level=0,
    ):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p

    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        eff_direction = eff_grad / (
            grad.abs().max(-1, keepdim=True)[0] + self.epsilon
        )
        return direction, eff_direction

    def forward(
        self,
        model,
        logits,
        input_ids_1,
        attention_mask_1,
        input_ids_2,
        attention_mask_2,
        dataset_name,
    ):
        # adv training
        if dataset_name not in ('similarity', 'sentiment', 'paraphrase'):
            raise NotImplementedError
        # init delta
        if dataset_name in ("similarity", "paraphrase"):
            input_embed_1 = model.bert.embed(input_ids_1)
            input_embed_2 = model.bert.embed(input_ids_2)
            noise_1 = generate_noise(input_embed_1, attention_mask_1, epsilon=self.noise_var)
            noise_2 = generate_noise(input_embed_2, attention_mask_2, epsilon=self.noise_var)
            noise = torch.cat([noise_1, noise_2], dim=1)
        else:
            input_embed_1 = model.bert.embed(input_ids_1)
            noise = generate_noise(input_embed_1, attention_mask_1, epsilon=self.noise_var)
        for step in range(0, self.K):
            if dataset_name == "similarity":
                noise_1, noise_2 = torch.split(noise, [input_embed_1.size(1), input_embed_2.size(1)], dim=1)
                adv_logits = model.predict_similarity_given_bert_input_embeds(
                    input_embed_1 + noise_1,
                    attention_mask_1,
                    input_embed_2 + noise_2,
                    attention_mask_2,
                )
                adv_loss = F.mse_loss(adv_logits, logits.detach(), reduction="sum")
            elif dataset_name == "paraphrase":
                noise_1, noise_2 = torch.split(noise, [input_embed_1.size(1), input_embed_2.size(1)], dim=1)
                adv_logits = model.predict_paraphrase_given_bert_input_embeds(
                    input_embed_1 + noise_1,
                    attention_mask_1,
                    input_embed_2 + noise_2,
                    attention_mask_2,
                )
                adv_loss = F.binary_cross_entropy_with_logits(adv_logits, logits.detach(), reduction="sum")
            else:
                adv_logits = model.predict_sentiment_given_bert_input_embeds(
                    input_embed_1 + noise,
                    attention_mask_1,
                )
                adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            (delta_grad,) = torch.autograd.grad(
                adv_loss, noise, only_inputs=True, retain_graph=False
            )
            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            noise, eff_noise = self._norm_grad(
                delta_grad, eff_grad=eff_delta_grad,
            )
            noise = noise.detach()
            noise.requires_grad_()
        if dataset_name == "similarity":
            adv_logits = model.predict_similarity_given_bert_input_embeds(
                input_embed_1 + noise_1,
                attention_mask_1,
                input_embed_2 + noise_2,
                attention_mask_2,
            )
            adv_loss = F.mse_loss(adv_logits, logits, reduction="mean")
        elif dataset_name == "paraphrase":
            adv_logits = model.predict_paraphrase_given_bert_input_embeds(
                input_embed_1 + noise_1,
                attention_mask_1,
                input_embed_2 + noise_2,
                attention_mask_2,
            )
            adv_loss = (
                F.binary_cross_entropy_with_logits(logits, F.sigmoid(adv_logits.detach()), reduction="mean") +
                F.binary_cross_entropy_with_logits(adv_logits, F.sigmoid(logits.detach()), reduction="mean")
            )
        else:
            adv_logits = model.predict_sentiment_given_bert_input_embeds(
                input_embed_1 + noise,
                attention_mask_1,
            )
            adv_loss = (
                F.kl_div(F.log_softmax(adv_logits, dim=1), F.softmax(logits.detach(), dim=1), reduction="batchmean") +
                F.kl_div(F.log_softmax(logits, dim=1), F.softmax(adv_logits.detach(), dim=1), reduction="batchmean")
            )
        return adv_loss
