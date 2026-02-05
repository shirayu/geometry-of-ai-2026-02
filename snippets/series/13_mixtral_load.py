from transformers import MixtralForCausalLM, AutoTokenizer
import torch

# Mixtral-8x7B のロード
model = MixtralForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
