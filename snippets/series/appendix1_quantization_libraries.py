# bitsandbytes（4-bit/8-bit量子化）
import bitsandbytes as bnb
linear_4bit = bnb.nn.Linear4bit(in_features, out_features)

# AutoGPTQ（GPTQ量子化）
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("model-gptq")

# llama.cpp（GGUF形式、CPU推論）
# コマンドライン: ./main -m model.gguf -p "prompt"
