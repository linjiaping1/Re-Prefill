import os
from typing import List, Optional, Union

import torch
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.utils import logging
from transformers.generation.utils import (
    SampleOutput,
    GenerationConfig,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput
)
from utils import select_key_visual_tokens


logger = logging.get_logger(__name__)


def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    # compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    compile_forward = False
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if self.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True
    is_prefill_2nd = is_prefill

    model_kwargs["img_key_position"] = {'image_start': 0, 'image_end': 0}
    model_kwargs["compute_attn_weight"] = True
    model_kwargs_rp = model_kwargs.copy()
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
            model_kwargs["compute_attn_weight"] = False
            model_kwargs_rp["compute_attn_weight"] = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        if is_prefill_2nd:  # for first token prediction
            input1_prefill_attn = list(outputs.attentions)
            vis_attention = torch.stack(input1_prefill_attn, dim=0)[:, :, 0, model_kwargs['img_key_position']["image_start"]:model_kwargs['img_key_position']["image_end"] + 1]
            vis_attention = torch.mean(vis_attention, dim=1)
            model_kwargs_rp["rp_s1_selected_token_idx"] = select_key_visual_tokens(vis_attention)
            model_kwargs_rp["rp_s1_selected_token_idx"] += model_kwargs['img_key_position']["image_start"]

            ## stage 1
            del outputs
            updated_logits, model_kwargs_rp, outputs = re_prefill(self, input_ids, model_kwargs_rp)
            ## stage 2 (use all vision tokens)
            input_seq_len = input_ids.shape[1]
            past_key_values_by_layer = model_kwargs['past_key_values'].layers
            inject_mask = torch.ones((past_key_values_by_layer[0].keys.shape[2],), dtype=torch.bool,
                                     device=input_ids.device)
            inject_mask[input_seq_len:] = False
            s2_token_idx = torch.arange(model_kwargs['img_key_position']["image_start"],
                                               model_kwargs['img_key_position']["image_end"] + 1) + input_seq_len
            inject_mask[s2_token_idx] = True

            outputs.attentions = list(outputs.attentions)
            if outputs.attentions is not None and len(outputs.attentions) == len(past_key_values_by_layer):
                for oa_i, out_attn in enumerate(outputs.attentions):
                    if out_attn is None:
                        break
                    outputs.attentions[oa_i] = out_attn[:, :, inject_mask.cpu()]
            for past_key_values in past_key_values_by_layer:
                past_key_values.keys = past_key_values.keys[:, :, inject_mask, :]
                past_key_values.values = past_key_values.values[:, :, inject_mask, :]
        else:
            updated_logits = next_token_logits

        is_prefill_2nd = is_prefill

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, updated_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (updated_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def re_prefill(self, input_ids, model_kwargs_rp):
    model_inputs_rp = self.prepare_inputs_for_generation(input_ids, **model_kwargs_rp)

    outputs_rp = self(**model_inputs_rp, return_dict=True)

    model_kwargs_rp = self._update_model_kwargs_for_generation(
        outputs_rp,
        model_kwargs_rp,
        is_encoder_decoder=self.config.is_encoder_decoder,
    )

    updated_logits = outputs_rp.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

    return updated_logits, model_kwargs_rp, outputs_rp
