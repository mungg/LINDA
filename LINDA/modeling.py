from transformers import BartConfig, BartPretrainedModel, BartModel
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

import numpy as np
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class LengthConverter(nn.Module):
    """
    Implementation of Length Transformation.
    """

    def __init__(self, args):
        super(LengthConverter, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(1., dtype=torch.float))
        self.args = args

    def forward(self, z, ls, z_mask):
        """
        Adjust the number of vectors in `z` according to `ls`.
        Return the new `z` and its mask.
        Args:
            z - latent variables, shape: B x L_x x hidden
            ls - target lengths, shape: B
            z_mask - latent mask, shape: B x L_x
        """
        n = z_mask.sum(1)  # represents lengths of input vs ls: represents targeted lengths
        arange_l = torch.arange(ls.max().long())
        arange_z = torch.arange(z.size(1))
        if torch.cuda.is_available():
            arange_l = arange_l.cuda()
            arange_z = arange_z.cuda()
        arange_l = arange_l[None, :].repeat(z.size(0), 1).float()  # bs, ls.max
        mu = arange_l * n[:, None].float() / ls[:, None].float()  # bs, ls.max
        arange_z = arange_z[None, None, :].repeat(z.size(0), ls.max().long(), 1).float()  # bs, ls.max, input_seq (128)

        distance = torch.clamp(arange_z - mu[:, :, None], -100, 100)  # bs, ls.max, 128
        logits = - torch.pow(2, distance) / (2. * self.sigma ** 2)  # bs, ls.max, 128

        logits = logits * z_mask[:, None, :] - 99. * (1 - z_mask[:, None, :])  # bs, ls.max, 128
        weight = torch.softmax(logits, 2)  # bs, ls.max, 128

        z_prime = (z[:, None, :, :] * weight[:, :, :, None]).sum(2)  # bs, ls.max, 768
        z_prime_mask = (arange_l < ls[:, None].float()).float()  # bs, ls.max
        z_prime = z_prime * z_prime_mask[:, :, None]  # bs, ls.max, 768
        return z_prime, z_prime_mask


class BartForLINDA(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # LINDA setting
        self.length_converter = LengthConverter(config)

        # for gaussian noise
        self.std=0.001
        self.mean=0

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min=1e-9)
    # todo: functions for LINDA
    def _convert_length(self, z, z_mask, target_lens):
        """Adjust the number of latent variables.
           [parameter description]
           z: input_ids
           z_mask: attention_ids
           target_lens: real_length of target (in our case, it is same as source's one)

           [return]
           converted vector with length of target

           [usage]
           z_with_y_length = self.convert_length(sampled_z, z_mask, y_mask.sum(-1))
        """
        converted_vectors, converted_vectors_mask = self.length_converter(z, target_lens, z_mask)
        return converted_vectors, converted_vectors_mask

    def get_noisy_embedding(self, input_ids_1=None, attention_mask_1=None,
                            input_ids_2=None, attention_mask_2=None, alpha=None, sent_emb=None, mix_style=None):

        alpha = torch.tensor(alpha).to(self.device)

        # into bart encoder
        encoded_inputs_1 = self.model.encoder(input_ids=input_ids_1, attention_mask=attention_mask_1)
        encoded_inputs_2 = self.model.encoder(input_ids=input_ids_2, attention_mask=attention_mask_2)

        encoded_input_1 = encoded_inputs_1.last_hidden_state
        encoded_input_2 = encoded_inputs_2.last_hidden_state

        # noisy embedding
        # get noisy input sentence first (noisy_input = alpha * sent_1 + (1-alpha) * sent_2)
        target_len = alpha * attention_mask_1.sum(1) + (1 - alpha) * attention_mask_2.sum(1)
        target_len = target_len.to(torch.int)

        # interpolation
        converted_input_1, converted_mask = self._convert_length(encoded_input_1, attention_mask_1, target_len)
        converted_input_2, converted_mask = self._convert_length(encoded_input_2, attention_mask_2, target_len)

        # check mask_1 == mask_2
        if mix_style:
            mu1 = converted_input_1.mean(dim=[1], keepdim=True)
            var1 = converted_input_1.var(dim=[1], keepdim=True)
            sig1 = (var1 + 1e-6).sqrt()
            mu1, sig1 = mu1.detach(), sig1.detach()
            input1_normed = (converted_input_1 - mu1) / sig1

            mu2 = converted_input_2.mean(dim=[1], keepdim=True)
            var2 = converted_input_2.var(dim=[1], keepdim=True)
            sig2 = (var2 + 1e-6).sqrt()

            mu2, sig2 = mu2.detach(), sig2.detach()

            mu_mix = alpha * mu1 + (1-alpha) * mu2
            sig_mix = alpha * var1 + (1-alpha) * var1

            noisy_input = input1_normed * sig_mix + mu_mix

        else:
            noisy_input = alpha * converted_input_1 + (1 - alpha) * converted_input_2

        output = BaseModelOutput(
            last_hidden_state=noisy_input, hidden_states=None, attentions=None
        )
        if sent_emb:
            return output, target_len, (converted_input_1, converted_input_2)
        else:
            return output, target_len

    def get_noisy_embedding_multiple(self, input_ids_list, attention_mask_list, alpha_list):
        encoded_inputs_list, sentence_embed_list = [], []
        target_len = 0
        for input_ids, attention_mask, alpha in zip(input_ids_list, attention_mask_list, alpha_list):
            encoded_inputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoded_inputs = encoded_inputs.last_hidden_state
            encoded_inputs_list.append(encoded_inputs)

            alpha = torch.tensor(alpha).to(self.device)
            target_len += alpha * attention_mask.sum(1)

        target_len = target_len.to(torch.int)

        noisy_input = None
        for encoded_inputs, attention_mask, alpha in zip(encoded_inputs_list, attention_mask_list, alpha_list):
            converted_input, converted_mask = self._convert_length(encoded_inputs, attention_mask, target_len)
            if noisy_input is None:
                noisy_input = alpha * converted_input
            else:
                noisy_input += alpha * converted_input

        output = BaseModelOutput(
            last_hidden_state=noisy_input, hidden_states=None, attentions=None
        )
        return output, target_len
    # todo: forward method
    def forward(
        self,
        input_ids=None, # previously, input_ids_1
        attention_mask=None, # previously, attention_mask_1
        masked_input_ids=None,
        masked_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None, # previously, label_1
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        L2_reg=0.001, # L2 loss omitted for now
        use_l2_loss=False,
        fix_pair=False,
        fix_alpha=False,
        add_gaussian_noise=True,
        alpha_for_beta=0.0,
    ):
        #### forward during training ####
        if self.training:

            # randomly draw an example from the batch
            if fix_pair:
                indexes = torch.range(0, input_ids.shape[0]).type(torch.int)
                indexes = torch.flip(indexes, dims=[0])
            else:
                indexes = torch.randperm(input_ids.shape[0])

            input_ids_2 = input_ids[indexes]
            attention_mask_2 = attention_mask[indexes]
            masked_input_ids_2 = None if masked_input_ids is None else masked_input_ids[indexes]
            masked_attention_mask_2 = None if masked_attention_mask is None else masked_attention_mask[indexes]
            labels_2 = labels[indexes]  # if inference, make label_2 None

            if fix_alpha:
                alpha = torch.tensor(0.5).to(self.device)
            elif alpha_for_beta > 0:
                alpha = torch.tensor(np.random.beta(alpha_for_beta, alpha_for_beta)).to(self.device)
                alpha = alpha.unsqueeze(0)
            else:
                alpha = torch.rand(1).to(self.device)

            # into bart encoder

            encoder_attention_1 = attention_mask if masked_attention_mask is None else masked_attention_mask
            encoder_attention_2 = attention_mask_2 if masked_attention_mask_2 is None else masked_attention_mask_2

            encoded_inputs_1 = self.model.encoder(input_ids=input_ids if masked_input_ids is None else masked_input_ids,
                                                  attention_mask=encoder_attention_1)
            encoded_inputs_2 = self.model.encoder(input_ids=input_ids_2 if masked_input_ids_2 is None else masked_input_ids_2,
                                                  attention_mask=encoder_attention_2)

            encoded_input_1 = encoded_inputs_1.last_hidden_state
            encoded_input_2 = encoded_inputs_2.last_hidden_state

            l2_loss = None
            if use_l2_loss:
                sent_1_l2 = torch.sum(torch.square(torch.linalg.norm(encoded_input_1, dim=-1)), dim=-1)
                sent_2_l2 = torch.sum(torch.square(torch.linalg.norm(encoded_input_2, dim=-1)), dim=-1)
                l2_loss_1 = torch.mean(sent_1_l2)
                l2_loss_2 = torch.mean(sent_2_l2)
                l2_loss = L2_reg * (alpha * l2_loss_1 + (1-alpha) * l2_loss_2)
                l2_loss = torch.unsqueeze(l2_loss, 0)
            if add_gaussian_noise:
                noisy_encoded_input_1 = encoded_input_1 + torch.randn_like(encoded_input_1) * self.std + self.mean
                noisy_encoded_input_2 = encoded_input_1 + torch.randn_like(encoded_input_2) * self.std + self.mean

            # get noisy input
            target_len = alpha * encoder_attention_1.sum(1) + (1 - alpha) * encoder_attention_2.sum(1)
            target_len = target_len.to(torch.int)

            converted_input_1, converted_mask = self._convert_length(noisy_encoded_input_1, encoder_attention_1, target_len)
            converted_input_2, converted_mask = self._convert_length(noisy_encoded_input_2, encoder_attention_2, target_len)

            noisy_input = alpha * converted_input_1 + (1 - alpha) * converted_input_2

            # into decoder
            outputs_1 = self.model.decoder(
                encoder_attention_mask=converted_mask,
                encoder_hidden_states=noisy_input,
                input_ids=shift_tokens_right(
                    input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
                ),
            )
            outputs_2 = self.model.decoder(
                encoder_attention_mask=converted_mask,
                encoder_hidden_states=noisy_input,
                input_ids=shift_tokens_right(
                    input_ids_2, self.config.pad_token_id, self.config.decoder_start_token_id
                ),
            )

            # compute loss
            lm_logits_1 = self.lm_head(outputs_1[0]) + self.final_logits_bias
            lm_logits_2 = self.lm_head(outputs_2[0]) + self.final_logits_bias

            loss_fct = CrossEntropyLoss()
            masked_lm_loss_1 = loss_fct(lm_logits_1.view(-1, self.config.vocab_size), labels.view(-1))
            masked_lm_loss_2 = loss_fct(lm_logits_2.view(-1, self.config.vocab_size), labels_2.view(-1))
            masked_lm_loss = alpha * masked_lm_loss_1 + (1 - alpha) * masked_lm_loss_2
            total_loss = masked_lm_loss if l2_loss is None else (masked_lm_loss+l2_loss)

            return total_loss, masked_lm_loss, l2_loss

        #### forward during eval (taken from BartForConditionalGeneration) ####
        else:
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if labels is not None:
                if decoder_input_ids is None:
                    decoder_input_ids = shift_tokens_right(
                        labels, self.config.pad_token_id, self.config.decoder_start_token_id
                    )

            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
