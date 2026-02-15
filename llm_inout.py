import sys
import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer, cache_utils

# --- path setup ---
sys.path.append('../../')

# --- local / custom imports ---
from aimet_utils.DotDict import DotDict, custom_nb_config
from utilities.profiler import event_marker
from huggingface.baseline_models.qwen2 import modeling_qwen2
from llm_utils.qc_adaptation import (
    QcAttention, 
    bypass_update_causal_mask, 
    MLP_prepare_conv, 
    ForCausalLM_prepare_conv, 
    MLP_forward_conv, 
    DynamicCache_update,
    DynamicCache_get_seq_length, 
    update_attr
)

# =============================================================================
# 1. Configuration Loading
# =============================================================================

config_path = Path('config/nb_config_tang.yml')
with open(config_path, 'r') as f:
    raw_cfg = yaml.safe_load(f)

nb_cfg = DotDict.from_dict(custom_nb_config(raw_cfg))

# Setting NSP Target and Model Params
htp_config_file = nb_cfg.model.htp_config_file
device = 'cuda'
ARN = nb_cfg.model.ARN
cache_dir = nb_cfg.model.cache_dir
output_dir = nb_cfg.output_dir

os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# 2. Model Adaptation (Patching)
# =============================================================================

with event_marker("FP model adaptation configuration"):
    # Attention Class Update
    modeling_qwen2.QWEN2_ATTENTION_CLASSES['eager'] = QcAttention

    # Bypass attention_mask preparation
    mask_update_success = update_attr(modeling_qwen2.Qwen2Model, '_update_causal_mask', bypass_update_causal_mask)
    mask_prepare_success = update_attr(modeling_qwen2.Qwen2Model, '_prepare_decoder_attention_mask', bypass_update_causal_mask)
    
    assert mask_update_success or mask_prepare_success, \
        f"Neither _prepare_decoder_attention_mask(..) nor _update_causal_mask(..) found. Unknown Qwen2Model definition in {modeling_qwen2.Qwen2Model}"

    # Adaptation to use Conv instead of Linear
    setattr(modeling_qwen2.Qwen2MLP, 'prepare_conv', MLP_prepare_conv)
    setattr(modeling_qwen2.Qwen2MLP, 'forward_conv', MLP_forward_conv)
    setattr(modeling_qwen2.Qwen2ForCausalLM, 'prepare_conv', ForCausalLM_prepare_conv)

    # Adapting KVS management
    assert update_attr(cache_utils.DynamicCache, 'update', DynamicCache_update), \
        f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
    assert update_attr(cache_utils.DynamicCache, 'get_seq_length', DynamicCache_get_seq_length), \
        f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"

# =============================================================================
# 3. Instantiate and Configure Model Definition
# =============================================================================

model_id = nb_cfg.model.model_id
llm_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)

# Context Length Configuration
# User can change this value (must be less than Qwen2 trained context length)
context_length = nb_cfg.model.context_length

# Debugging: Reduce layers if profiling is enabled
if nb_cfg.profiling.qk_layer:
    llm_config.num_hidden_layers = 2

print(f'num_layer: {llm_config.num_hidden_layers}, context_length : {context_length},'
      f'num_hidden_size :{llm_config.num_attention_heads}, num_kv_heads: {llm_config.num_key_value_heads}')

# =============================================================================
# 4. Fixed Settings (Do Not Change)
# =============================================================================
# Auto-regression settings: strictly enforced for downstream consumption

fixed_attributes = {
    'return_top_k': 0,
    'return_new_key_value_only': True,
    'transposed_key_cache': True,
    'use_combined_mask_input': True,
    'use_position_embedding_input': True,
    'use_cache': True,
    '_attn_implementation': 'eager',
    '_attn_implementation_internal': 'eager',
    'use_input_embeddings': True,
    'mask_neg': nb_cfg.model.mask_neg,
    'rm_div': True
}

for attr, value in fixed_attributes.items():
    setattr(llm_config, attr, value)

# Final Name Formatting
model_name = os.path.basename(model_id).lower()
model_name = model_name.replace(".", "p").replace("-", "_")


def slice_inputs_and_run_successive_kvcache_inference_custom(batch_id, fpm, input_ids=None, input_embeds=None, args=None, **kwargs):
    if input_ids is not None:
        input_length = input_ids.shape[1]
    else:
        input_length = input_embeds.shape[1]

    outputs = {}
    attention_mask = kwargs.pop('attention_mask', None)

    cnt = 0
    for idx in range(0, input_length, fpm.num_tokens)[::-1]:
        if cnt >= args.eval_token:
            break
        
        idx = input_length - idx

        if attention_mask is not None:
            cache_offset = attention_mask.shape[1] - input_length
            kwargs["attention_mask"] = attention_mask[:, max(0, cache_offset + idx - fpm.max_tokens):cache_offset + idx]

        if input_ids is not None:
            cur_outputs = fpm(input_ids=input_ids[:, max(0, idx - fpm.num_tokens):idx], **kwargs)
        elif input_embeds is not None:
            prepared_inputs, kvcache_info_bundle = fpm.prepare_inputs(input_ids=None, input_embeddings=input_embeds[:, max(0, idx - fpm.num_tokens):idx, :], **kwargs)
            outputs_step = fpm.model(**prepared_inputs)

            save_dir = args.save_path
            os.makedirs(save_dir, exist_ok=True)
            torch.save(prepared_inputs, save_dir + f"inputs_b{batch_id}_t{cnt}.pt")
            torch.save(outputs_step, save_dir + f"outputs_b{batch_id}_t{cnt}.pt")

            cur_outputs = fpm.prepare_outputs(outputs_step, prepared_inputs, kvcache_info_bundle)
            cnt += 1

            # cur_outputs = fpm(input_ids=None, input_embeddings=input_embeds[:, max(0, idx - fpm.num_tokens):idx, :],
            #                   **kwargs) ### this is another equivalent method to generate cur_outputs but slower as it calls fpm again
        else:
            print("No input_ids or input_embeds provided to inference generator!")
            assert False

        # get valid outputs
        bsz, length, dim = cur_outputs['lm_logits'].shape

        outputs['lm_logits'] = torch.cat(
            (outputs.get('lm_logits', torch.zeros((bsz, 0, dim), device=fpm.device)), cur_outputs['lm_logits']),
            dim=1)
        kwargs['past_key_values'] = outputs['past_key_values'] = cur_outputs['past_key_values']

    return outputs

def generate_inout(model_mode, data_loader, forward_pass_manager, num_batches=0, args=None):

    if num_batches == 0:
        num_batches = len(data_loader)
    loss = 0

    for batch_id, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Evaluating")):
        if batch_id >= num_batches:
            break
        if model_mode == "kvcache":
            outputs = slice_inputs_and_run_successive_kvcache_inference_custom(batch_id, forward_pass_manager, input_embeds=batch['input_embeddings'], args=args)
        elif model_mode == "bertcache":
            outputs = forward_pass_manager(**batch)
        # outputs = slice_inputs_and_run_successive_kvcache_inference(forward_pass_manager, input_embeds=batch['inputs_embeds'])

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--return_dict", action="store_false", help="return dict by default")
    parser.add_argument("--eval_batch", default=10, help='Number of batches generated.', required=False, type=int)
    parser.add_argument("--eval_token", default=10, help='Number of tokens generated for each batch.', required=False, type=int)
    parser.add_argument("--save_path", default='outputdir/', help='Dir to save.', required=False, type=str)
    args = parser.parse_args()

    with event_marker('FP model'):
        model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(model_id, config=llm_config)
        # model.config.return_dict = False
        model.config.return_dict = args.return_dict ##### this is set to be true to compare the results with quantized model outputs!!! #####
        os.environ['TOKENIZERS_PARALLELISM'] = '0'
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
        ## Adjust the tokenizer to limit to context_length
        tokenizer.model_max_length = context_length

    with event_marker('FP model adaptation for NSP backend completion'):
        for name, module in model.named_modules():
            if hasattr(module, "prepare_conv"):
                module.prepare_conv()

    # Loading the calibration data from notebook config
    if nb_cfg.calib.name == 'json':
        # device = "cuda:0"
        device = "cpu"
        from llm_utils.qwen2_vl_dataloader import get_qwen2_dataset
        qwen2_dataset_setting = {
            "emb_length": ARN,
            "device": device,
            "qwen2vl_model_id": nb_cfg.model.model_id,
            "calibration_dataset_path": nb_cfg.calib.calibration_dataset_path,
            "ppl_evaluation_dataset_path": nb_cfg.calib.ppl_evaluation_dataset_path,
            "image_dataset_path": nb_cfg.calib.image_dataset_path,
            "vision_input_size": nb_cfg.calib.vision_input_size
        }
        train_dataloader, test_dataloader, dataset = get_qwen2_dataset(model.model, qwen2_dataset_setting, num_test_batches=100)

    elif nb_cfg.calib.name == 'wiki':
        from llm_utils.wikitext_dataloader import get_wiki_dataset
        train_dataloader, test_dataloader, _ = get_wiki_dataset(context_length, tokenizer, cache_dir)
    else:
        raise RuntimeError("Invalid dataset setting from notebook config")


    # ---
    # ### 4. Generate input and output
    from torch.nn import CrossEntropyLoss
    # from llm_utils.forward_pass_wrapper import slice_inputs_and_run_successive_kvcache_inference

    # ### 4.1 FP32 PPL Eval
    from llm_utils.forward_pass_wrapper import LLMForwardPassManager

    orig_fpm = LLMForwardPassManager(cfg=llm_config, model=model, tokenizer=tokenizer,
                                     model_mode='kvcache', num_logits_to_return=ARN, separate_tuple_input_output=False,
                                     num_tokens=ARN)

    with event_marker("FP eval"):
        with torch.no_grad():
            with orig_fpm.place_on_device(device):
                 _ = generate_inout('kvcache', test_dataloader, orig_fpm, num_batches=args.eval_batch, args=args)

    print("Processing completed!")

