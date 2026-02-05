# bitsandbytes（4-bit/8-bit量子化）
import bitsandbytes as bnb
from auto_gptq import AutoGPTQForCausalLM

in_features = 1024
out_features = 1024

linear_4bit = bnb.nn.Linear4bit(in_features, out_features)

# AutoGPTQ（GPTQ量子化）
model = AutoGPTQForCausalLM.from_quantized("model-gptq")

# llama.cpp（GGUF形式、CPU推論）
# コマンドライン: ./main -m model.gguf -p "prompt"
