import pandas as pd
import json
import re
import sys
import os
import argparse

def getArgs():
     #获取参数
    parser = argparse.ArgumentParser(description="Compute mechanism penalty with configurable parameters.")
    # parser.add_argument('--model', type=str, required=True, help='Model name or path, e.g., Qwen/Qwen2.5-7B-Instruct')
    # parser.add_argument('--train_type', type=str, default='lora', help='Training type, e.g., lora')
    parser.add_argument('--liar_file', type=str, required=True, help='liar_file')
    parser.add_argument('--output_file', type=str, required=True, help='output_file')    
    parser.add_argument('--output_file_2', type=str, required=True, help='output_file_2')    

args = getArgs()
 
# 读取LIAR数据集
liar_file = args.liar_file
# liar_file = 'valid.tsv'

liar_data = pd.read_csv(liar_file, sep='\t', header=None)

# 定义输出文件
output_file = args.output_file
output_file_2 = args.output_file_2

# 打开输出文件
with open(output_file, 'w') as jsonl_file:
    # 遍历LIAR数据集的每一行
    for index, row in liar_data.iterrows():
        statement_id = row[0]
        label = row[1]
        statement = row[2]
        subject = row[3]
        speaker = row[4]
        speaker_job_title = row[5]
        state_info = row[6]
        party_affiliation = row[7]
        context = row[13]

        # 根据条件替换标签
        if label in ['false', 'pants-fire', 'barely-true']:
            label = 'false'
        elif label in ['mostly-true', 'true']:
            label = 'true'
        elif label == 'half-true':
            continue  # 跳过这一条数据

        # 构造Qwen数据集格式
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"Please evaluate the truthfulness of the news. statement:'{statement}', subject:'{subject}', speaker:'{speaker}', speaker_job_title:'{speaker_job_title}', party_affiliation:'{party_affiliation}', context:'{context}'."  # 用户提出的问题或陈述
                },
                {
                    "from": "gpt",
                    # "value": f"This statement is {label}."
                    "value": f"{label}"

                }
            ],
            "system": "Please evaluate the truthfulness of the statement."
            # ,"rejected_response": "I don't know."
        }

        # 将转换结果写入jsonl文件
        jsonl_file.write(json.dumps(conversation) + '\n')

print(f"转换完成，结果已保存至 {output_file}")

#过滤脏数据
def filter_jsonl(input_file, output_file, key_to_remove='json'):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            record = json.loads(line)
            # 如果记录不包含指定的键，则写入输出文件
            if not re.search(r'\.json', line):
                outfile.write(json.dumps(record) + '\n')

# 使用示例
filter_jsonl(output_file, output_file_2)
# filter_jsonl(output_file, 'binary_valid_data_final.jsonl')

print(f"转换完成，结果已保存至 {output_file}")