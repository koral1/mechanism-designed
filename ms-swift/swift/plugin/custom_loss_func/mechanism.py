# 机制设计：
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


def compute_distance(
      output: torch.Tensor,
      target: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      distance_method: str = 'l2'
) -> torch.Tensor:
    # 检查输出和目标的形状
    """计算输出与目标之间的距离"""
    if distance_method == 'l2':
        distance = torch.norm(output - target, p=2, dim=-1)
    elif distance_method == 'l1':
        distance = torch.norm(output - target, p=1, dim=-1)
    elif distance_method == 'cosine':
        distance = 1 - F.cosine_similarity(output, target, dim=-1)
    else:
        raise ValueError(f"Unknown distance type: {distance_method}")

    if attention_mask is not None:
        distance = distance * attention_mask

    return distance

def compute_mechanism_penalty(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        distance_method: str = 'l2',
        lambda_penalty: float = 1,  # 惩罚系数
        min_distance_threshold: float = 0.1  # 最小距离阈值
) -> torch.Tensor:
    """计算机制惩罚项"""
    distance = compute_distance(outputs, targets, attention_mask, distance_method)

    
    # 仅在距离超过阈值时施加惩罚
    penalty = torch.where(
        distance > min_distance_threshold,
        lambda_penalty * (distance - min_distance_threshold),
        torch.zeros_like(distance)
    )

    
    if attention_mask is not None:
        penalty = (penalty * attention_mask).sum() / attention_mask.sum()
    else:
        penalty = penalty.mean()
    
    print(f"distance:{distance},lambda_penalty:{lambda_penalty},penalty:{penalty}")

        
    return penalty