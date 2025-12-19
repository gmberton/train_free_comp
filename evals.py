
import re
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

def get_num(text):
    # Try finding number after ####, otherwise grab the last number found
    nums = re.findall(r"####\s*(-?\d+\.?\d*)", text.replace(",", "")) or re.findall(r"-?\d+\.?\d*", text)
    return int(float(nums[-1])) if nums else None

@torch.no_grad()
def eval_gsm8k(model, tok, num_samples=100, bsz=32):
    tok.padding_side = "left"
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(list(range(0, len(ds), max(1, len(ds) // num_samples)))[:num_samples])  # Subsample evenly
    correct = 0
    for i in tqdm(range(0, len(ds), bsz), desc="Eval GSM8K", leave=False):
        batch = ds[i:i+bsz]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": q + "\nBe brief and output final answer after '####'."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False
            ) for q in batch['question']
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.amp.autocast(device_type=str(model.device), dtype=model.dtype):
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False, top_p=None, top_k=None, temperature=None)
        # Slice output to remove input tokens
        str_predictions = tok.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        num_predictions = [get_num(pred) for pred in str_predictions]
        ground_truths = [int(d.split('#### ')[1].replace(',', '')) for d in batch['answer']]
        correct += (np.array(num_predictions) == np.array(ground_truths)).sum()
    acc = correct / len(ds)
    return round(acc * 100, 1)

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    llm_name = "Qwen/Qwen3-4B"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    print(eval_gsm8k(llm, tokenizer))

