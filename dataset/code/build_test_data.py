import pandas as pd
import numpy as np
import re

# 读取tsv文件
df = pd.read_csv('test.tsv', sep='\t', header=None)

# 定义列名
columns = ['ID', 'label', 'statement', 'subject', 'speaker', 'speaker_job_title', 'state_info', 'party_affiliation',
           'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']

# 重命名DataFrame的列
df.columns = columns

# 将NaN替换为空字符串或特定标记，以便后续处理
df.replace({np.nan: ''}, inplace=True)  # 或者使用其他标记，例如'unknown'

# 清理文本中的多余空格及不可见字符
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text).strip()  # 替换连续的空白字符为单个空格
        return text
    else:
        return text

for col in df.columns:
    df[col] = df[col].apply(clean_text)

# 构造新的prompt和label列
# 假设 df 是你的 DataFrame
# 检查并填充缺失值
df['statement'] = df['statement'].fillna('')
df['subject'] = df['subject'].fillna('')
df['speaker'] = df['speaker'].fillna('')
df['speaker_job_title'] = df['speaker_job_title'].fillna('')
df['party_affiliation'] = df['party_affiliation'].fillna('')
df['context'] = df['context'].fillna('')

# 使用 try-except 来处理可能出现的错误
def create_prompt(row):
    try:
        return f"Please evaluate the truthfulness of the news. statement:'{row['statement']}', subject:'{row['subject']}', speaker:'{row['speaker']}', speaker_job_title:'{row['speaker_job_title']}', party_affiliation:'{row['party_affiliation']}', context:'{row['context']}' ."
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return None  # 或者返回一个默认值

# 应用函数
df['prompt'] = df.apply(create_prompt, axis=1)
df['label'] = df.apply(lambda row: f"{row['label']}", axis=1)

# 定义一个函数来重新分类标签
def reclassify_label(label):
    if label in ['false', 'pants-fire', 'barely-true']:
        return 'false'
    elif label in ['mostly-true', 'true']:
        return 'true'
    else:
        # 如果有未知或其他标签，可以选择保留原样或进行其他处理
        return None

# 使用列表推导式重新分类所有标签
reclassified_labels = [reclassify_label(label) for label in df['label']]

# 将重新分类后的标签放回 DataFrame 中
df['label'] = reclassified_labels

# 移除标签为None的行
filtered_df = df[df['label'].notna()]

filtered_df = filtered_df[~filtered_df['prompt'].str.contains('json')]


# 选择需要的列并保存为csv文件
result_df = filtered_df[['prompt', 'label']]

result_df.to_csv('binary_test_data.csv', index=False, encoding='utf-8')

print("转换完成，结果已保存至 binary_test_data.csv")