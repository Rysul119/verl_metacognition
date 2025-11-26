"""
Utilities for processing and evaluating intermediate reasoning states.
This module provides functionality for extracting reasoning states from 
model responses, appending candidate answers, and computing metrics between
predicted and ground truth reasoning paths.
"""

import re
from copy import deepcopy
from typing import List, Tuple, Dict
import torch
from tensordict import TensorDict # need to install
from torch.nn import functional as F

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

def extract_accumulated_thoughts(
    response_text: str, 
    split_by_periods: bool = True,
    split_by_newlines: bool = True
) -> List[str]:
    """
    Extract accumulated from a model response.
    
    This function splits a response text into a sequence of reasoning thoughts,
    returning each progressive thoughts in the reasoning process.
    
    Args:
        response_text (str): The full response text to extract reasoning from
        split_by_periods (bool): Whether to split reasoning steps by periods
        split_by_newlines (bool): Whether to split reasoning steps by newlines
        
    Returns:
        List[str]: A list of accumulated thoughts to help build reasoning states
    """
    # Get individual reasoning steps
    thoughts = []
    
    # Apply both splitting methods if enabled
    if split_by_periods and split_by_newlines:
        # First split by newlines
        newline_splits = re.split(r'\n+', response_text)
        # Then split each segment by periods
        for segment in newline_splits:
            period_splits = re.split(r'\.(?!\d)', segment)  # Split by periods not followed by digits
            for step in period_splits:
                if step.strip():  # Only include non-empty steps
                    thoughts.append(step.strip() + ".")
    elif split_by_periods:
        # Split by periods not followed by digits (to preserve numbers like 3.14)
        steps = re.split(r'\.(?!\d)', response_text)
        thoughts = [step.strip() + "." for step in steps if step.strip()]
    elif split_by_newlines:
        # Split by newlines
        steps = re.split(r'\n+', response_text)
        thoughts = [step.strip() for step in steps if step.strip()]
    
    # Construct accumulated reasoning thoughts to help build reasoning states
    
    accumulated_thoughts = []
    # for the reasoning states list after discarding the last thought
    for i in range(len(thoughts)-1):
        current_intermediate = " ".join(thoughts[:i+1])
        accumulated_thoughts.append(current_intermediate)
    
    if len(accumulated_thoughts)>0:
        return accumulated_thoughts
    else:
        return [response_text]


def process_input_for_actor(
    data,
    candidate: str,
    tokenizer,
    max_response_length: int
) -> Tuple[DataProto, List[int]]:
    
    """
    Process input for the actor's _forward_micro_batch method.
    This function takes data a DataProtoItem
    Builds the reasoning states and append candidate to each reasoning state.
    Goes through the appended reasoning states encode with tokenizer and right pad them using pad_2d_list_to_length
    Creates a properly formatted DataProto object with all necessary 
    tensors for the actor model.

    Args:
    a DataProtoItem
    - prompts: [prompt_length], prompt token ids from dataset.
    - responses: [response_length], output token ids include response tokens from LLM generation and observation tokens from tool_calls.
    - response_mask: [response_length], 1 for LLM generated tokens, 0 for observation/padding tokens. But not used
    - input_ids: [prompt_length + response_length], whole sequence token ids, including prompt tokens and response tokens.
    - attention_mask: [prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
    - position_ids: [prompt_length + response_length], incremental position ids.
    """
    
    prompt_ids = data.batch["prompts"]
    eos_token_id = tokenizer.eos_token_id
    prompt_length = prompt_ids.size(0)

    response_ids = data.batch["responses"]
    valid_response_length = data.batch["attention_mask"][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length] # as right padded

    # decode
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    position_ids = data.batch["position_ids"] # of prompt + response
    attention_mask = data.batch["attention_mask"] # of prompt + response
    #ground_truth = data.non_tensor_batch["reward_model"]["ground_truth"]

    reasoning_states = extract_accumulated_thoughts(response_str)

    # append_candidate_to_reasoning states
    reasoning_states_candidate = [] # encoded list
    separator = "\nAnswer: " # not sure if just a white space or this
    reasoning_state_lengths = []
    for state in reasoning_states:
        encoded_state= tokenizer.encode(state, add_special_tokens=False)
        reasoning_state_lengths.append(len(encoded_state))
        encoded_candidate = tokenizer.encode(separator + candidate, add_special_tokens=False)
        reasoning_states_candidate.append(encoded_state + encoded_candidate)

    # right pad
    reasoning_states_candidate = pad_2d_list_to_length(reasoning_states_candidate, tokenizer.pad_token_id, max_length=max_response_length).to(prompt_ids.device)
    
    batch_size = reasoning_states_candidate.size(0)
    # preprend prompt_ids to the reasoning states + candidate
    seq = torch.cat([prompt_ids.repeat(batch_size, 1), reasoning_states_candidate], dim=-1) # dim batch, prompt+max_response_length
    reasoning_states_candidate_length = reasoning_states_candidate.size(1)
    delta_position_id = torch.arange(1, reasoning_states_candidate_length + 1, device=position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
    if position_ids.dim() == 3:  # qwen2vl mrope
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

    response_position_ids = position_ids[prompt_length-1: prompt_length].repeat(batch_size, 1) + delta_position_id # prompt's last position id + delta 
    position_ids = torch.cat([position_ids[:prompt_length].repeat(batch_size, 1), response_position_ids], dim=-1)
    response_attention_mask = get_response_mask(
        response_id=reasoning_states_candidate, eos_token=eos_token_id, dtype=attention_mask.dtype
    )
    attention_mask = torch.cat((attention_mask[:prompt_length].repeat(batch_size, 1), response_attention_mask), dim=-1)

    # all the tp ranks should contain the same data here. data in all ranks are valid
    batch = TensorDict(
        {
            "prompts": prompt_ids.repeat(batch_size, 1),
            "responses": reasoning_states_candidate, # dim (num_states, max_response length)
            "input_ids": seq,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )
    tensors = {
            "prompts": prompt_ids.repeat(batch_size, 1),
            "responses": reasoning_states_candidate, # dim (num_states, max_response length)
            "input_ids": seq,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    return DataProto.from_dict(tensors=tensors, auto_padding=True), reasoning_state_lengths # use the reasoning state lengths to slice over the log_probs

def calc_distance(
    gt: str,
    pred: str,
    extra_info: Dict
) -> torch.Tensor:
    
    """
    calculate the sum of log probs of candidate for given reasoning states
    """
    actor = extra_info["actor"] 
    data = extra_info["data"] 
    tokenizer = extra_info["tokenizer"]
    max_response_length = extra_info["max_response_length"]

    data_gt, state_lens = process_input_for_actor(data, gt, tokenizer, max_response_length)
    o = actor(data_gt)
    log_probs = o.batch["log_probs"]
    s_log_probs_gt = [-log_probs[i, s_len:].sum() for i, s_len in enumerate(state_lens)] # slicing to acquire sum(log_probs) for the candidate

    data_pred, state_lens = process_input_for_actor(data, pred, tokenizer, max_response_length)
    o = actor(data_pred)
    log_probs = o.batch["log_probs"]
    s_log_probs_pred = [-log_probs[i, s_len:].sum() for i, s_len in enumerate(state_lens)] # slicing to acquire sum(log_probs) for the candidate

    distance = torch.stack([torch.tensor(s_log_probs_pred), torch.tensor(s_log_probs_gt)], dim=0)

    return distance.T


def calc_distance_bak(
    gt: str,
    pred: str,
    extra_info: Dict
) -> torch.Tensor:
    
    """
    calculate the sum of log probs of candidate for given reasoning states
    """
    actor = extra_info["actor"] 
    temperature = extra_info["temperature"] 
    data = extra_info["data"] 
    tokenizer = extra_info["tokenizer"]
    max_response_length = extra_info["max_response_length"]

    data_gt, state_lens = process_input_for_actor(data, gt, tokenizer, max_response_length)
    with torch.no_grad():
        _, log_probs = actor._forward_micro_batch(data_gt, temperature=temperature, calculate_entropy=False)

    s_log_probs_gt = [-log_probs[i, s_len:].sum() for i, s_len in enumerate(state_lens)] # slicing to acquire sum(log_probs) for the candidate

    data_pred, state_lens = process_input_for_actor(data, pred, tokenizer, max_response_length)
    with torch.no_grad():
        _, log_probs = actor._forward_micro_batch(data_pred, temperature=temperature, calculate_entropy=False)

    s_log_probs_pred = [-log_probs[i, s_len:].sum() for i, s_len in enumerate(state_lens)] # slicing to acquire sum(log_probs) for the candidate

    distance = torch.stack([torch.tensor(s_log_probs_pred), torch.tensor(s_log_probs_gt)], dim=0)

    return distance.T

def extract_prediction(
    data_source: str,
    response: str
):
    if data_source == "openai/gsm8k":
        from . import gsm8k
        
        return gsm8k.extract_solution(solution_str=response)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "gneubig/aime-1983-2024", "mrk21/math"]:
        try:
            string_in_last_boxed = last_boxed_only_string(response)
            if string_in_last_boxed is not None:
                return remove_boxed(string_in_last_boxed)
            else:
                return string_in_last_boxed
        except Exception as e:
            print(f"Found exception: {e}")

def calc_coherence_anchor_drift(
    data_source: str,
    response: str,
    ground_truth: str,
    extra_info
):  
    # worst values
    coherence = 0. 
    anchor = 1.
    drift = 0.
    
    prediction = extract_prediction(data_source, response)
    if prediction is not None:
        distance = calc_distance(ground_truth, prediction, extra_info)
        T = distance.shape[0] # number of states
        # condition: closer to GT than to prediction
        closer_to_gt = distance[:, 1] <= distance[:, 0]
        # Coherence
        coherence = closer_to_gt.float().mean().item()
        
        if closer_to_gt.any():
            # Anchor (first index satisfying condition)
            anchor = closer_to_gt.float().nonzero(as_tuple=False).min().item() / T
            # Drift (last index satisfying condition)
            drift = closer_to_gt.float().nonzero(as_tuple=False).max().item() / T
        
    return coherence, anchor, drift

def calc_coherence_anchor_drift_bak_bak_bak_bak(
    data_source: str,
    response: str,
    ground_truth: str,
    extra_info
):  
    # worst values
    coherence = 0. 
    anchor = 1.
    drift = 0.
    if data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "gneubig/aime-1983-2024", "mrk21/math"]:
        try:
            string_in_last_boxed = last_boxed_only_string(response)
            if string_in_last_boxed is not None:
                prediction = remove_boxed(string_in_last_boxed)
                distance = calc_distance(ground_truth, prediction, extra_info)
                T = distance.shape[0] # number of states
                # condition: closer to GT than to prediction
                closer_to_gt = distance[:, 1] <= distance[:, 0]
                # Coherence
                coherence = closer_to_gt.float().mean().item()
                # Anchor (first index satisfying condition)
                if closer_to_gt.any():
                    anchor = closer_to_gt.float().nonzero(as_tuple=False).min().item() / T
                # Drift (last index satisfying condition)
                if closer_to_gt.any():
                    drift = closer_to_gt.float().nonzero(as_tuple=False).max().item() / T
        except Exception as e:
            print(f"Found exception: {e}")
        
        return coherence, anchor, drift

def calc_coherence_anchor_drift_bak_bak_bak(
    data_source: str,
    response: str,
    ground_truth: str,
    extra_info
):  
    # worst values
    coherence = 0. 
    anchor = 1.
    drift = 0.
    if data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "gneubig/aime-1983-2024", "mrk21/math"]:
        string_in_last_boxed = last_boxed_only_string(response)
        if string_in_last_boxed is not None:
            prediction = remove_boxed(string_in_last_boxed)
            distance = calc_distance(ground_truth, prediction, extra_info)
            T = distance.shape[0] # number of states
            # condition: closer to GT than to prediction
            closer_to_gt = distance[:, 1] <= distance[:, 0]
            # Coherence
            coherence = closer_to_gt.float().mean().item()
            # Anchor (first index satisfying condition)
            if closer_to_gt.any():
                anchor = closer_to_gt.float().nonzero(as_tuple=False).min().item() / T
            # Drift (last index satisfying condition)
            if closer_to_gt.any():
                drift = closer_to_gt.float().nonzero(as_tuple=False).max().item() / T
        
        return coherence, anchor, drift

def calc_coherence_anchor_drift_bak(
    data_source: str,
    response: str,
    ground_truth: str,
    extra_info
):
    # worst values
    coherence = 0.
    anchor = 1.
    drift = 0.
    if data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "gneubig/aime-1983-2024", "mrk21/math"]:
        try:
            string_in_last_boxed = last_boxed_only_string(response)
            if string_in_last_boxed is not None:
                prediction = remove_boxed(string_in_last_boxed)
                distance = calc_distance(ground_truth, prediction, extra_info)
                T = distance.shape[0] # number of states
                # condition: closer to GT than to prediction
                closer_to_gt = distance[:, 1] <= distance[:, 0]
                # Coherence
                coherence = closer_to_gt.float().mean().item()
                # Anchor (first index satisfying condition)
                if closer_to_gt.any():
                    anchor = closer_to_gt.float().nonzero(as_tuple=False).min().item() / T
                # Drift (last index satisfying condition)
                if closer_to_gt.any():
                    drift = closer_to_gt.float().nonzero(as_tuple=False).max().item() / T

        except Exception as e:
            print(f"Found exception: {e}")

        return coherence, anchor, drift


def calc_coherence_anchor_drift_bak_bak(
    data_source: str,
    response: str,
    ground_truth: str,
    extra_info
):  
    if data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "gneubig/aime-1983-2024", "mrk21/math"]:
        prediction = remove_boxed(last_boxed_only_string(response))
    
    distance = calc_distance(ground_truth, prediction, extra_info)
    T = distance.shape[0] # number of states

    # condition: closer to GT than to prediction
    closer_to_gt = distance[:, 1] <= distance[:, 0]

    # Coherence
    coherence = closer_to_gt.float().mean().item()

    # Anchor (first index satisfying condition)
    if closer_to_gt.any():
        anchor = closer_to_gt.float().nonzero(as_tuple=False).min().item() / T
    else:
        anchor = 1.0    # worst value
    # Drift (last index satisfying condition)
    if closer_to_gt.any():
        drift = closer_to_gt.float().nonzero(as_tuple=False).max().item() / T
    else:
        drift = 0.0     # worst value

    return coherence, anchor, drift


