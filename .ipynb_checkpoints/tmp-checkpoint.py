from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.add_tokens(["true", "false"])  # 新增标记的索引从`vocab_size`开始
print(tokenizer("true")["input_ids"])  # 查看TRUE的索引
print(tokenizer("false")["input_ids"])  # 查看TRUE的索引