import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 读取验证数据
df = pd.read_csv('dataset/data/test_data.csv')
prompts = df['prompt'].tolist()
true_labels = df['label'].tolist()

#机制设计-binary-lambda
tokenizer = AutoTokenizer.from_pretrained("output/mechanism_similarity/lambda-0-val/v2-20250316-013213/checkpoint-500-merged")

# 设置采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# 初始化LLM
llm = LLM(model="output/mechanism_similarity/lambda-0-val/v2-20250316-013213/checkpoint-500-merged")

predicted_labels = []
generated_texts = []

for prompt in prompts:
    messages = [
        {"role": "system", "content": "Please evaluate the truthfulness of the statement."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    outputs = llm.generate([text], sampling_params)
    
    for output in outputs:
        generated_text = output.outputs[0].text.strip().lower()  # 获取生成的文本并清理
        # if 'false' in generated_text:
        #     predicted_label = 'false'
        # elif 'half-true' in generated_text:
        #     predicted_label = 'half-true'
        # elif 'pants-fire' in generated_text:
        #     predicted_label = 'pants-fire'
        # elif 'barely-true' in generated_text:
        #     predicted_label = 'barely-true'
        # elif 'mostly-true' in generated_text:
        #     predicted_label = 'mostly-true'
        # elif 'true' in generated_text:
        #     predicted_label = 'true'
        # else:
        #     predicted_label = None  # 如果没有找到任何标签，则设为None
        if 'false' in generated_text or 'pants-fire' in generated_text or 'barely-true' in generated_text:
            predicted_label = 'False'
        elif 'mostly-true' in generated_text or 'true' in generated_text:
            predicted_label = 'True'
        else:
            predicted_label = None  # 如果没有找到任何标签，则设为None

        
        predicted_labels.append(predicted_label)
        generated_texts.append(generated_text)

true_labels = [str(label) for label in true_labels]
predicted_labels = [str(label) for label in predicted_labels]

# 将预测标签转换为列表格式，以便于比较
predicted_labels = [l if l is not None else 'unknown' for l in predicted_labels]  # 处理未识别的情况

# 计算各项指标
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 输出详细的分类报告
report = classification_report(true_labels, predicted_labels)
print(report)
# 将报告保存到文件
with open('report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")

    f.write(report)

# print("报告已保存到 'sft_classification_report.txt'")
print("报告已保存到 'report.txt'")


df['predicted_label'] = predicted_labels
df['generated_text'] = generated_texts
# df.to_csv('sft_test_results_with_predictions.csv', index=False)
df.to_csv('results_with_predictions.csv', index=False)