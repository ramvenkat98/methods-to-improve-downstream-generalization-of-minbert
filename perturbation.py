# This file is taken from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/perturbation.py
# in the MT-DNN repository (associated with the SMART paper: https://arxiv.org/abs/1911.03437)
# and is originally authored by the SMART authors.
# We adapt it to our use case here in order to use the smoothness-inducing regularization
# technique.
import torch
import logging
import torch.nn.functional as F


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
        if dataset_name not in ('similarity', 'sentiment'):
            raise NotImplementedError
        # init delta
        if dataset_name == "similarity":
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
            else:
                adv_logits = model.predict_sentiment_given_bert_input_embeds(
                    input_embed_1 + noise,
                    attention_mask_1,
                )
            adv_loss = F.mse_loss(adv_logits, logits.detach(), reduction="sum")
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
        else:
            adv_logits = model.predict_sentiment_given_bert_input_embeds(
                input_embed_1 + noise,
                attention_mask_1,
            )
        adv_loss = F.mse_loss(logits, adv_logits, reduction="mean")
        return adv_loss
