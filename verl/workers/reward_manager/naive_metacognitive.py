from collections import defaultdict

import torch
import numpy as np

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

@register("naive_metacognitive")
class NaiveMetaCognitiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, weights, actor, config, return_dict=False):
        """
        actor: actor from the actor work group to get the hidden_states
        """

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        data.meta_info["_verl_auto_padding"] = True # pad when batch size is not divisible by chunk size
        mean_residuals = actor(data, config.reward_model.metacognition.hidden_reduction)  # (B, L, H)
        mhs = torch.permute(mean_residuals.batch["hidden_states"], (1,0,2)).to(dtype=torch.float32)[config.reward_model.metacognition.hlayers]     # dim (L, B, H) 
        bs = mhs.shape[1] # batch size as _verl_auto_padding create some padded input
        #print("naive_metacognitive mhs shape {} weights shape {}".format(mhs.shape, weights.shape))
        activations = torch.matmul(mhs, weights) # mean_hidden_states dim (L,B, H) weights dim (L, H, 1), activation dim (L, B, 1)
        thresholds = torch.median(activations, dim=1)[0] # (L, 1)
        #print("data size {} activations dim {} thresholds dim {}".format(len(data), activations.shape, thresholds.shape))
        already_print_data_sources = {}

        for i in range(bs):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            extra_info["activations"] = activations[:, i, :]  # (L, 1) 
            extra_info["thresholds"] = thresholds # (L, 1)
            extra_info["data"] = data_item
            extra_info["tokenizer"] = self.tokenizer
            extra_info["max_response_length"] = config.data.max_response_length
            extra_info["return_hidden"] = True

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
