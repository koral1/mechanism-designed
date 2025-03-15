#安装swift
# git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
export HF_ENDPOINT=https://hf-mirror.com


#更新依赖
#SFT
pip install --upgrade pip
pip install jinja2 peft
pip install transformers -U
python -m bitsandbytes
pip install transformers==4.38.0
pip install markupsafe==2.0.1
pip install --upgrade pydantic
pip install --upgrade deepspeed
pip install accelerate==1.0.1
pip install vllm
# conda install -c conda-forge mlx-lm
# pip install mlx-lm
pip install --upgrade pyarrow
pip install transformers==4.40.0
pip install bitsandbytes -U
conda update --all




#RLHF

#dpo
conda update --all
conda install libgcc libstdcxx-ng
pip install transformers -U
conda remove pandas
conda install pandas
pip install typing_extensions
pip install markupsafe==2.0.1
pip install accelerate==0.34.2
pip install transformers==4.42.0
pip install --upgrade pyarrow
pip install --upgrade huggingface_hub

# 复制
sudo cp /opt/conda/pkgs/libstdcxx-14.2.0-hc0a3c3a_1/lib/libstdc++.so.6.0.33 /opt/conda/lib/python3.8/site-packages/aistudio_common/reader/libs/libstdc++.so.6
# 删除之前链接
sudo rm /opt/conda/lib/python3.8/site-packages/aistudio_common/reader/libs/libstdc++.so.6
# 创建新的链接
sudo ln -s /opt/conda/pkgs/libstdcxx-14.2.0-hc0a3c3a_1/lib/libstdc++.so.6.0.33 /opt/conda/lib/python3.8/site-packages/aistudio_common/reader/libs/libstdc++.so.6



# 下载完模型后
pip install transformers==4.40.0
pip install torch transformers scikit-learn

#sft推理/合并lora
pip install vllm
pip install --upgrade jinja2
pip install transformers -U
pip install torch transformers scikit-learn


#huggingface
pip install -U huggingface_hub