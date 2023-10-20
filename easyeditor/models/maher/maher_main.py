import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn  
import torch.optim as optim 
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

# from .compute_ks import compute_ks
# from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .maher_hparams import MAHERHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

# q = ' , question: Who is the president of the US?'
# qc= 'info: The current president of the United States is Donald Trump, question: Who is the president of the US?'



def apply_maher_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MAHERHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    print("MAHER is HEREEEEEE!!!")
    # weights_copy = {}
    # if copy:
    #     model = deepcopy(model)

    weights_copy = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        ).detach().clone()
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    # weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(requests)
    # import pdb
    # pdb.set_trace()
    
    base_q = f'''q2: {requests[0]['prompt']}, Answer to q2: '''
    qc = f'''q1: {requests[0]['prompt']}, Answer to q1:  {requests[0]['target_new']}
    q2: {requests[0]['prompt']}, Answer to q2: '''
    
    qc_out = run_model(model, qc, tok)
    qc_out_decoded = tok.decode(qc_out[0])

    qc_answer = qc_out_decoded.split(base_q)[1]
    qc_answer_without_target = qc_answer.split(requests[0]['target_new'])[0]

    q = base_q + qc_answer_without_target
    q_out = run_model(model, q, tok)

    print([(i, tok.decode(x)) for i, x in enumerate(q_out)])
    print([(i, tok.decode(x)) for i, x in enumerate(qc_out)])

    print("Q ############# Before:", [tok.decode(x) for x in q_out])
    print("C ############# Before:", [tok.decode(x) for x in qc_out])

    token_index = find_first_disagreement(q_out, qc_out, len(tok.encode(q)), len(tok.encode(qc)), tok)
    print(token_index)
    max_tokens_modified = 3
    while token_index != (-1, -1) and max_tokens_modified>0:
        print("index", token_index)
        update_model_at_index(model, token_index, q, qc, tok, hparams=hparams)
        q_out = run_model(model, q, tok)
        qc_out = run_model(model, qc, tok)

        print("Q ############# After:", [tok.decode(x) for x in [q_out]])
        print("C ############# After:", [tok.decode(x) for x in [qc_out]])

        token_index = find_first_disagreement(q_out, qc_out, len(tok.encode(q)), len(tok.encode(qc)), tok)
        max_tokens_modified -= 1
        # break


    return model, weights_copy
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_memit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(f"cuda:{hparams.device}"), val_mat.to(f"cuda:{hparams.device}")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def cosine_similarityy(vec1, vec2):  
    dot_product = np.dot(vec1, vec2)  
    magnitude_vec1 = np.linalg.norm(vec1)  
    magnitude_vec2 = np.linalg.norm(vec2)  
    similarity = dot_product / (magnitude_vec1 * magnitude_vec2)  
    return similarity  


    
def update_layer(layer, key, value, i, use_c=False):
    layer = layer
    if False:
        print("using")
        u = key        
        u = get_inv_cov(
            model,
            tok,
            "model.layers.{}.mlp.down_proj".format(i),
            "wikipedia",
            100000,
            "float32",
            hparams=hparams,
        ) @ u.unsqueeze(1)
        u = u.squeeze()
    else:
        u = key
    left = u
    left = left / left.norm()
    with torch.no_grad():
        cur_value = layer(key)
    diff = cosine_similarityy(value.cpu().numpy(), cur_value.cpu().numpy())
    print("values difference", diff)
    if True:
#     if diff < 0.95:
        print("modifing layer")    
        right = (value - cur_value) / torch.dot(key, left)
        upd_matrix = (left.unsqueeze(1) @ right.unsqueeze(0))        
        w = layer.weight.data + upd_matrix.T
        layer.weight.data = w
        return w
    return False


def find_first_disagreement(q_out, qc_out, skip_count_q_out, skip_count_qc_out, tok):  
    min_length = min(len(q_out) - skip_count_q_out, len(qc_out) - skip_count_qc_out)  
#     return (53, 65)
    for i in range(min_length):  
        if q_out[i + skip_count_q_out] != qc_out[i + skip_count_qc_out]:
            print(tok.decode(q_out[i + skip_count_q_out]), tok.decode(qc_out[i + skip_count_qc_out]))
            return (i + skip_count_q_out -1 , i + skip_count_qc_out -1)  
  
    # If no disagreements found, return (-1, -1)  
    return (-1, -1)  



signals_k = []
signals_v = []

def run_model(model, prompt, tok=None):
#     correct_prompts = [
#     prompt,
#     ]
    batch = tok(prompt, return_tensors='pt')
    with torch.no_grad():
        post_edit_outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask=batch['attention_mask'].to('cuda'),
            do_sample=False,

        #     max_length=15
            max_new_tokens=30
        )
#     print(post_edit_outputs.detach().cpu().numpy().tolist()[0][len(batch["input_ids"][0]):])
#     print(tok.decode(post_edit_outputs.detach().cpu().numpy().tolist()[0][len(batch["input_ids"][0]):]))
    return post_edit_outputs.detach().cpu().numpy().tolist()[0]

m2q = []
h2q = []
h3qc = []
softmax = []

def get_kv_for_layer(layer_i, model, token_index=(0,0), q=None, qc=None, tok=None):
#     global signals_k, signals_v
#     signals_k = []
#     signals_v = []
    print(f"getting k and v for the layer number {layer_i}")
    global m2q, h2q, h3qc, softmax
    layer = model.get_submodule(f"model.layers.{layer_i}")
    post_attention_layernorm = layer.post_attention_layernorm
    down_proj = layer.mlp.down_proj
    
    m2q = []
    h2q = []
    h3qc = []
    softmax = []
    
    def get_m2q_from_down_proj(module, inp, output):
        global m2q
        m2q += [inp[0][0]]
        return output
    
    def get_h2q_from_post_attention_layernorm(module, inp, output):
        global h2q
        h2q += [inp[0][0]]
        return output
    
    def get_h3qc_from_layer(module, inp, output):
        global h3qc
        h3qc += [output[0][0]]
        return output
    
    def get_softmax_prob(module, inp, output):
        global softmax
#         print("")
#         print(output)
        softmax += [output[0]]
#         print(len(softmax))
        return output

    m2q_hook = down_proj.register_forward_hook(get_m2q_from_down_proj)
    h2q_hook = post_attention_layernorm.register_forward_hook(get_h2q_from_post_attention_layernorm)
    softmax_hook = model.lm_head.register_forward_hook(get_softmax_prob)
    
    q_out = run_model(model, q, tok)
    m2q_hook.remove()
    h2q_hook.remove()
    print(f"Q", [tok.decode(x) for x in [q_out]])
    
    
    q_logits = softmax
    softmax = []
    h3qc_hook = layer.register_forward_hook(get_h3qc_from_layer)
    qc_out = run_model(model, qc, tok)
    h3qc_hook.remove()
    softmax_hook.remove()
#     print(len(qc_out))
#     out = run_model(model, 'info: trump, question: Who is the president of the united states?')
    print(f"QC", [tok.decode(x) for x in [qc_out]])

    # if find_first_disagreement(q_out, qc_out, len(tok.encode(q)), len(tok.encode(qc)), tok) != token_index:
#         return False
    
    qc_logits = softmax
    m2q = torch.cat(m2q)
    h2q = torch.cat(h2q)
    h3qc = torch.cat(h3qc)
    q_logits = torch.cat(q_logits)
    qc_logits = torch.cat(qc_logits)
#     print(q_logits.shape)
#     print(qc_logits.shape)
#     print(q_logits[2].shape)
    
    k = m2q[token_index[0]]
    v = h3qc[token_index[1]] - h2q[token_index[0]] 
    # v = h3qc[token_index[1]] 
     
    
    return k, v, q_logits[token_index[0]], qc_logits[token_index[1]]


probs = []
def update_model_at_index(model, token_index, q, qc, tok=None, hparams=None):
    global probs
    for i in hparams.layers:
        layer = model.model.layers[i]
        KV = get_kv_for_layer(i, model, token_index, q, qc, tok)
        if KV:
            K, V, q_logits, qc_logits = KV
            update_layer(layer.mlp.down_proj, K, V, i)
            # generate_heatmap_image(q,qc, tok, model, f"heatmaps/heatmap_modified_layer_{i}.png")
        else:
            print("current Token Index changed so looking for other one")
            break

        


























# def execute_memit(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     requests: List[Dict],
#     hparams: MAHERHyperParams,
#     cache_template: Optional[str] = None,
# ) -> Dict[str, Tuple[torch.Tensor]]:
#     """
#     Executes the MEMIT update algorithm for the specified update at the specified layer
#     Invariant: model at beginning of function == model at end of function
#     """

#     deltas = {}

#     # Update target and print info
#     requests = deepcopy(requests)
#     for i, request in enumerate(requests):
#         if request["target_new"][0] != " ":
#             # Space required for correct tokenization
#             requests[i]["target_new"] = " " + request["target_new"]

#         if '{}' not in request['prompt']:
#             assert request['subject'] in request['prompt'] or \
#                    print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

#             requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

#     for request in requests[:10]:
#         print(
#             f"MEMIT request sample: "
#             f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
#         )

#     # Retrieve weights that user desires to change
#     weights = {
#         f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
#             model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
#         )
#         for layer in hparams.layers
#     }
#     # Save old weights for future restoration
#     weights_copy = {k: v.detach().clone() for k, v in weights.items()}

#     # Compute z for final layer
#     context_templates = get_context_templates(model, tok)
#     z_layer = hparams.layers[-1]
#     z_list = []

#     for request in requests:
#         # Retrieve k/v pair if already stored in cache
#         cache_fname = (
#             Path(
#                 str(cache_template).format(
#                     z_layer, hparams.clamp_norm_factor, request["case_id"]
#                 )
#             )
#             if cache_template is not None
#             else None
#         )
#         data_loaded = False
#         if (
#             cache_fname is not None  # Require cache template
#             and cache_fname.exists()  # Cache file must exist
#         ):
#             try:
#                 data = np.load(cache_fname)
#                 z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
#                 data_loaded = True
#             except Exception as e:
#                 print(f"Error reading cache file due to {e}. Recomputing...")

#         # Compute k/v pair if not loaded from cache
#         if not data_loaded:
#             cur_z = compute_z(
#                 model,
#                 tok,
#                 request,
#                 hparams,
#                 z_layer,
#                 context_templates,
#             )

#             z_list.append(cur_z)

#             if cache_fname is not None:
#                 cache_fname.parent.mkdir(exist_ok=True, parents=True)
#                 np.savez(
#                     cache_fname,
#                     **{
#                         "v_star": cur_z.detach().cpu().numpy(),
#                     },
#                 )
#                 print(f"Cached k/v pair at {cache_fname}")
#     zs = torch.stack(z_list, dim=1)

#     # Insert
#     for i, layer in enumerate(hparams.layers):
#         print(f"\n\nLAYER {layer}\n")

#         # Get current model activations
#         layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
#         print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

#         # Compute residual error
#         cur_zs = get_module_input_output_at_words(
#             model,
#             tok,
#             z_layer,
#             context_templates=[request["prompt"] for request in requests],
#             words=[request["subject"] for request in requests],
#             module_template=hparams.layer_module_tmp,
#             fact_token_strategy=hparams.fact_token,
#             track='out'
#         ).T
#         targets = zs - cur_zs
#         print("z error", torch.linalg.norm(targets, dim=0).mean())

#         repeat_factor = (layer_ks.size(1) // targets.size(1))
#         targets = targets.repeat_interleave(repeat_factor, dim=1)

#         # Load covariance matrix
#         force_recompute = False
#         # force_recompute = layer != hparams.layers[0]
#         cov = get_cov(
#             model,
#             tok,
#             hparams.rewrite_module_tmp.format(layer),
#             hparams.mom2_dataset,
#             hparams.mom2_n_samples
#             if not force_recompute
#             else hparams.mom2_n_samples // 10,
#             hparams.mom2_dtype,
#             force_recompute=force_recompute,
#             hparams=hparams
#         )

#         # Compute update in double precision
#         layer_ks, targets = (
#             layer_ks.double(),
#             targets.double(),
#         )

#         adj_k = torch.linalg.solve(
#             hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
#             layer_ks,
#         )
#         resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
#         upd_matrix = resid @ adj_k.T

#         # Adjust update matrix shape
#         weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
#         upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

#         print("orig norm", torch.linalg.norm(weights[weight_name]))
#         print("upd norm", torch.linalg.norm(upd_matrix))

#         # Update model weights and record desired changes in `delta` variable
#         with torch.no_grad():
#             weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
#             deltas[weight_name] = (
#                 adj_k.detach().cpu(),
#                 resid.detach().cpu(),
#             )

#         # Clear GPU memory
#         cov.cpu()
#         for x in [layer_ks, cur_zs, targets]:
#             x.cpu()
#             del x
#         torch.cuda.empty_cache()

#     # Restore state of original model
#     with torch.no_grad():
#         for k, v in weights.items():
#             v[...] = weights_copy[k]

#     print(f"Deltas successfully computed for {list(weights.keys())}")

#     return deltas


# def get_cov(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     layer_name: str,
#     mom2_dataset: str,
#     mom2_n_samples: str,
#     mom2_dtype: str,
#     inv: bool = False,
#     force_recompute: bool = False,
#     hparams=None,
# ) -> torch.Tensor:
#     """
#     Retrieves covariance statistics, then computes the algebraic inverse.
#     Caches result for future use.
#     """

#     model_name = model.config._name_or_path.replace("/", "_")
#     key = (model_name, layer_name)

#     print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
#     if key not in COV_CACHE or force_recompute:
#         stat = layer_stats(
#             model,
#             tok,
#             layer_name,
#             hparams.stats_dir,
#             mom2_dataset,
#             to_collect=["mom2"],
#             sample_size=mom2_n_samples,
#             precision=mom2_dtype,
#             hparams=hparams,
#             force_recompute=force_recompute,
#         )
#         COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

#     return (
#         torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}")) if inv else COV_CACHE[key].to(f"cuda:{hparams.device}")
#     )


# def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
#     """
#     GPT-2 and GPT-J have transposed weight representations.
#     Returns a matrix that matches the desired shape, else raises a ValueError
#     """

#     if matrix.shape == shape:
#         return matrix
#     elif matrix.T.shape == shape:
#         return matrix.T
#     else:
#         raise ValueError(
#             "Update matrix computed by MEMIT does not match original weight shape. "
#             "Check for bugs in the code?"
#         )


# def get_context_templates(model, tok):
#     global CONTEXT_TEMPLATES_CACHE

#     if CONTEXT_TEMPLATES_CACHE is None:
#         CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
#             [
#                 f.replace("{", " ").replace("}", " ") + ". {}"
#                 for f in generate_fast(
#                     model,
#                     tok,
#                     ["The", "Therefore", "Because", "I", "You"],
#                     n_gen_per_prompt=n_gen // 5,
#                     max_out_len=length,
#                 )
#             ]
#             for length, n_gen in [(10, 5)]  # Be careful about changing this.
#         ]
#         print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

#     return CONTEXT_TEMPLATES_CACHE
