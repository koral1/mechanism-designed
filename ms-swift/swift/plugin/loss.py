# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Callable, Optional

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import sys
import os
import argparse
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# 假设这是你的根目录
root_dir = 'mechanism-designed/ms-swift'
sys.path.append(os.path.join(root_dir, 'swift', 'plugin'))

from custom_loss_func.mechanism import compute_mechanism_penalty
from custom_loss_func.simple import simple_mechanism_loss_compute


class LossType:
    loss_scale = 'loss_scale'
    mechanism = 'mechanism'
    simple_mechanism_loss = 'simple_mechanism_loss'
    mechanism_similarity = 'mechanism_similarity'

LOSS_MAPPING = {}

def getArgs():
     #获取参数
    parser = argparse.ArgumentParser(description="Compute mechanism penalty with configurable parameters.")
    parser.add_argument('--model', type=str, required=True, help='Model name or path, e.g., Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--train_type', type=str, default='lora', help='Training type, e.g., lora')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help='Batch size per device during training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Alpha for LoRA')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Number of gradient accumulation steps')
    parser.add_argument('--eval_steps', type=int, default=100, help='Number of steps between evaluations')
    parser.add_argument('--save_steps', type=int, default=100, help='Number of steps between saving checkpoints')
    parser.add_argument('--save_total_limit', type=int, default=2, help='Maximum number of checkpoints to save')
    parser.add_argument('--logging_steps', type=int, default=5, help='Number of steps between logging')
    parser.add_argument('--model_author', type=str, default='swift', help='Author of the model')
    parser.add_argument('--model_name', type=str, default='swift-robot', help='Name of the model')
    parser.add_argument('--loss_type', type=str, default='mechanism', help='Type of loss, e.g., mechanism')
    parser.add_argument('--distance_method', type=str, default='l2', help='Distance computation method, e.g., l2 (same as --method)')
    parser.add_argument('--lambda_value', type=float, default=0.5, help='Lambda value for the loss function')
    parser.add_argument('--min_distance_threshold', type=float, default=0.1, help='Minimum distance threshold for applying penalty')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the results')
    parser.add_argument('--base_lambda', type=float, help='Base_lambda value')
    parser.add_argument('--torch_dtype', type=str, default='float32', help='The data type for the model (e.g., bfloat16, float32)')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Batch size for evaluation per device')
    parser.add_argument('--target_modules', type=str, default='all-linear', help='Target modules for fine-tuning')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of input sequences')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='Number of workers for the data loader')

    args = parser.parse_args()
    return args

# 计算文本与标签之间的余弦相似度
def cosine_distance(text1, text2, vectorizer):
    tfidf_matrix = vectorizer.transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return 1 - cosine_sim[0][0]  # 返回 1 - 相似度

def register_loss_func(loss_type: str, loss_func: Optional[Callable] = None):
    loss_info = {}

    if loss_func is not None:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_type] = loss_info
        return

    def _register_loss_func(loss_func: Callable) -> Callable:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_type] = loss_info
        return loss_func

    return _register_loss_func


def ce_loss_func(outputs, labels):
    logits = outputs.logits
    device = logits.device
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].to(device)
    # Save memory
    masks = shift_labels != -100
    shift_logits = shift_logits[masks]
    shift_labels = shift_labels[masks]
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits, shift_labels)
    return loss, masks

#解码
def decode_tokens(token_ids, tokenizer):
    """
    将 token IDs 转换为文本。
    :param token_ids: Tensor 或 List[int]，token IDs
    :param tokenizer: Hugging Face Tokenizer
    :return: 解码后的文本
    """
    # 转换为列表（如果输入是张量）
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    # 解码 token IDs
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text

# 计算交叉熵损失，并将 outputs 和 labels 转换为文本
def ce_loss_func_with_decoding(outputs, labels, tokenizer):
    logits = outputs.logits
    device = logits.device
    
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].to(device)
    
    # Save memory
    masks = shift_labels != -100
    shift_logits = shift_logits[masks]
    shift_labels = shift_labels[masks]
    
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits, shift_labels)
    
    # 解码 outputs 和 labels
    predicted_token_ids = shift_logits.argmax(dim=-1)  # 模型预测的 token IDs
    decoded_outputs = [decode_tokens(ids, tokenizer) for ids in predicted_token_ids]
    decoded_labels = [decode_tokens(ids, tokenizer) for ids in shift_labels]
    # print(f"decoded_outputs:{decoded_outputs},decoded_labels:{decoded_labels},loss:{loss},outputs:{outputs},labels:{labels}")
    
    return loss, masks, decoded_outputs, decoded_labels


# 基于语义相似度的机制设计损失
@register_loss_func(LossType.mechanism_similarity)
def mechanism_similarity_loss_func(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    """Loss func

    Args:
        outputs: The model outputs
        labels: The labels
        loss_scale: The loss scale
        num_items_in_batch: Number of tokens in the labels of gradient accumulation round that are not -100.

    Returns:

    """
   
    args = getArgs()
    # distance_method = args.distance_method
    lambda_value = args.lambda_value
    # min_distance_threshold = args.min_distance_threshold
    # print(f"distance_method:{distance_method},lambda_value:{lambda_value},min_distance_threshold:{min_distance_threshold}")

    # lambda_value = 10

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #交叉熵损失函数
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    loss_origin, masks, decoded_outputs, decoded_labels = ce_loss_func_with_decoding(outputs, labels, tokenizer)
    # print(f"loss_origin:{loss}\n")

    distances = []
    output_sentence = ''.join(decoded_outputs).strip()
    label_sentence = ''.join(decoded_labels).strip()
    # print(f"output_sentence:{output_sentence},label_sentence:{label_sentence}\n")

    #初始化vectorizer
    vectorizer = TfidfVectorizer()
    # 拟合并转换文本
    tfidf_matrix = vectorizer.fit_transform([output_sentence, label_sentence])

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    distances.append(1-cosine_sim)
    # print(f"distances:{distances}")

    # 计算机制设计损失
    distance_penalty = lambda_value * torch.tensor(distances).float().to(device)
    loss = loss_origin + distance_penalty

    # print(f"loss_final:{loss}")

    # 保存结果到 CSV
    # 当前仅lambda =0时保存（？）
    # data = {
    #     'loss_origin': [loss_origin.item()],
    #     'loss_final': [loss.item()],
    #     'cosine_sim': [cosine_sim],
    #     'distance': [distance]
    # }
    # df = pd.DataFrame(data)

    # # 保存到 CSV 文件
    # csv_file_path = "loss_results.csv"
    # df.to_csv(csv_file_path, mode='a', header=not pd.io.common.file_exists(csv_file_path), index=False)

    
    loss = loss.mean()
    return loss

# Use @register_loss_func to decorate your own loss, use --loss_type xxx to train
@register_loss_func(LossType.loss_scale)
def loss_scale_func(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    """Loss func

    Args:
        outputs: The model outputs
        labels: The labels
        loss_scale: The loss scale
        num_items_in_batch: Number of tokens in the labels of gradient accumulation round that are not -100.

    Returns:

    """
    loss, masks = ce_loss_func(outputs, labels)
    if loss_scale is not None:
        shift_scale = loss_scale[..., 1:].to(masks.device)
        shift_scale = shift_scale[masks]
        loss = (shift_scale * loss)
    if num_items_in_batch is None:
        loss = loss.mean()
    else:
        # compat transformers>=4.46
        loss = loss.sum() / num_items_in_batch
    return loss


# Use @register_loss_func to decorate your own loss, use --loss_type xxx to train
#机制设计
@register_loss_func(LossType.mechanism)
def mechanism_loss_func(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    """Loss func

    Args:
        outputs: The model outputs
        labels: The labels
        loss_scale: The loss scale
        num_items_in_batch: Number of tokens in the labels of gradient accumulation round that are not -100.

    Returns:

    """
   
    args = getArgs()
    distance_method = args.distance_method
    lambda_value = args.lambda_value
    min_distance_threshold = args.min_distance_threshold
    # print(f"distance_method:{distance_method},lambda_value:{lambda_value},min_distance_threshold:{min_distance_threshold}")

    #交叉熵损失函数
    logits = outputs.logits
    device = logits.device
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].to(device)
    # Save memory
    masks = shift_labels != -100
    shift_logits = shift_logits[masks]
    shift_labels = shift_labels[masks]
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits, shift_labels)
    print(f"loss_origin:{loss}\n")

    #加上机制设计损失函数
    loss = loss+ compute_mechanism_penalty(F.softmax(shift_logits, dim=-1), F.one_hot(shift_labels,shift_logits.shape[1]),None,distance_method,lambda_value,min_distance_threshold)
    print(f"loss_final:{loss}")
    
    loss = loss.mean()
    return loss

#简单机制设计--暂未使用
@register_loss_func(LossType.simple_mechanism_loss)
def simple_mechanism_loss_func(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    """Loss func

    Args:
        outputs: The model outputs
        labels: The labels
        loss_scale: The loss scale
        num_items_in_batch: Number of tokens in the labels of gradient accumulation round that are not -100.

    Returns:

    """
   
    args = getArgs()
    distance_method = args.distance_method
    min_distance_threshold = args.min_distance_threshold
    base_lambda = args.base_lambda

    # print(f"distance_method:{distance_method},lambda_value:{lambda_value},min_distance_threshold:{min_distance_threshold}")

    #交叉熵损失函数
    logits = outputs.logits
    device = logits.device
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].to(device)
    # Save memory
    masks = shift_labels != -100
    shift_logits = shift_logits[masks]
    shift_labels = shift_labels[masks]
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits, shift_labels)

    #加上机制设计损失函数
    loss = loss + simple_mechanism_loss_compute(F.softmax(shift_logits, dim=-1), F.one_hot(shift_labels,shift_logits.shape[1]), None,distance_method , min_distance_threshold, base_lambda)
    # + compute_mechanism_penalty(F.softmax(shift_logits, dim=-1), F.one_hot(shift_labels,shift_logits.shape[1]),None,distance_method,lambda_value,min_distance_threshold)

    loss = loss.mean()
    return loss


def get_loss_func(loss_type: Optional[str]) -> Optional[Callable]:
    if loss_type is None:
        return None
    return LOSS_MAPPING[loss_type]['loss_func']