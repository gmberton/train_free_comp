

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = 'cuda'
llm_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True, torch_dtype=torch.float32).to(device)
print(llm(input_ids=torch.tensor([[1000, 2000]]).to(device)).logits.shape)
# output: torch.Size([1, 2, 151936])
print(len(tokenizer))
# output: 151669
print(tokenizer.vocab_size)
# output: 151643

