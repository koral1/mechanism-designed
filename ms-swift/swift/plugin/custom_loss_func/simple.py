# 机制设计代码
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from swift import Swift, SwiftConfig, SwiftModel


def compute_lambda_min(outputs, targets, min_distance_threshold, base_lambda ):
    """计算理论保证的λ_min"""
    # 计算最大可能偏差
    delta_max = torch.max(torch.abs(outputs - targets))
    # 计算最小距离
    d_min = min_distance_threshold
    # λ_min = Δ_max / d_min
    return delta_max / d_min if d_min > 0 else base_lambda

def simple_mechanism_loss_compute(outputs, targets, attention_mask=None,distance_method:str ="l2" , min_distance_threshold:float=0.1, base_lambda:float=1):
    # 计算理论λ_min
    lambda_min = self.compute_lambda_min(outputs, targets,min_distance_threshold, base_lambda)
    print(f"lambda_min:{lambda_min},base_lambda:{base_lambda}")
    # 使用max确保λ不小于理论最小值
    lambda_actual = max(base_lambda, lambda_min)

    # 计算距离
    if distance_method == 'l2':
        distance = torch.norm(outputs - targets, p=2, dim=-1)
    elif method == 'l1':
        distance = torch.norm(output - target, p=1, dim=-1)
    elif method == 'cosine':
        distance = 1 - F.cosine_similarity(output, target, dim=-1)
    else:
        raise ValueError(f"Unknown distance type: {method}")

    # 计算惩罚
    penalty = lambda_actual * torch.where(
        distance > min_distance,
        distance - min_distance,
        torch.zeros_like(distance)
    )

    return penalty.mean()