"""
Utilities for metacognitive reward calculations. -+(a-theta) mean over layers
"""

import torch

def get_metacognitive_reward(ground_truth, extra_info):
    sign = 1.0 if ground_truth==1 else -1
    return sign * torch.mean(extra_info['activations'] - extra_info['thresholds']).item()
