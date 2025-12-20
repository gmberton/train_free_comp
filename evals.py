
import re
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

MAX_NEW_TOKENS = 256

def get_num(text):
    # Try finding number after ####, otherwise grab the last number found
    nums = re.findall(r"####\s*(-?\d+\.?\d*)", text.replace(",", "")) or re.findall(r"-?\d+\.?\d*", text)
    return int(float(nums[-1])) if nums else None

def get_bool(text):
    # Parse True/False (or yes/no) after ####, otherwise grab the last one found
    s = text.strip().lower()
    m = re.findall(r"####\s*(true|false|yes|no)\b", s)
    if not m:
        m = re.findall(r"\b(true|false|yes|no)\b", s)
    if not m:
        return None
    val = m[-1]
    return 1 if val in ("true", "yes") else 0

def get_choice_1_2(text):
    # Parse 1/2 after ####, otherwise grab the last 1/2 found
    s = text.strip()
    m = re.findall(r"####\s*([12])\b", s)
    if not m:
        m = re.findall(r"\b([12])\b", s)
    return int(m[-1]) if m else None

def get_sentiment(text):
    # Parse Positive/Negative after ####, otherwise grab the last one found
    s = text.strip().lower()
    m = re.findall(r"####\s*(positive|negative)\b", s)
    if not m:
        m = re.findall(r"\b(positive|negative)\b", s)
    if not m:
        return None
    return 1 if m[-1] == "positive" else 0

def _prep_tokenizer(tokenizer):
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def _subsample_evenly(ds, num_samples):
    if num_samples is None or num_samples >= len(ds):
        return ds
    step = max(1, len(ds) // num_samples)
    idx = list(range(0, len(ds), step))[:num_samples]
    return ds.select(idx)

@torch.no_grad()
def compute_preds(model, tokenizer, questions):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        ) for q in questions
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.amp.autocast(device_type=model.device.type, dtype=model.dtype):
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, top_p=None, top_k=None, temperature=None)
    # Slice output to remove input tokens
    return tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def eval_gsm8k(model, tokenizer, num_samples=100, bsz=32):
    _prep_tokenizer(tokenizer)
    ds = load_dataset("gsm8k", "main", split="test")
    ds = _subsample_evenly(ds, num_samples)
    correct = 0
    for i in tqdm(range(0, len(ds), bsz), desc="Eval GSM8K", ncols=100, leave=False):
        batch = ds[i:i+bsz]
        questions = [q + "\nBe brief and output final answer after '####'." for q in batch['question']]
        str_predictions = compute_preds(model, tokenizer, questions)
        num_predictions = [get_num(pred) for pred in str_predictions]
        ground_truths = [int(d.split('#### ')[1].replace(',', '')) for d in batch['answer']]
        correct += (np.array(num_predictions) == np.array(ground_truths)).sum()
    acc = correct / len(ds)
    return round(acc * 100, 1)

@torch.no_grad()
def eval_boolq(model, tokenizer, num_samples=100, bsz=32):
    _prep_tokenizer(tokenizer)
    ds = load_dataset("super_glue", "boolq", split="validation")
    ds = _subsample_evenly(ds, num_samples)
    correct = 0
    for i in tqdm(range(0, len(ds), bsz), desc="Eval BoolQ", ncols=100, leave=False):
        batch = ds[i:i+bsz]
        questions = [
            f"Passage:\n{p}\n\nQuestion: {q}\nAnswer True or False. Be brief and output final answer after '####'."
            for p, q in zip(batch["passage"], batch["question"])
        ]
        str_predictions = compute_preds(model, tokenizer, questions)
        bool_preds = [get_bool(p) for p in str_predictions]
        correct += (np.array(bool_preds) == np.array(batch["label"])).sum()
    return round((correct / len(ds)) * 100, 1)

@torch.no_grad()
def eval_sst2(model, tokenizer, num_samples=100, bsz=32):
    _prep_tokenizer(tokenizer)
    ds = load_dataset("glue", "sst2", split="validation")
    ds = _subsample_evenly(ds, num_samples)
    correct = 0
    for i in tqdm(range(0, len(ds), bsz), desc="Eval SST2", ncols=100, leave=False):
        batch = ds[i:i+bsz]
        questions = [
            f"Sentence: {s}\nClassify sentiment as Positive or Negative. Be brief and output final answer after '####'."
            for s in batch["sentence"]
        ]
        str_predictions = compute_preds(model, tokenizer, questions)
        sent_preds = [get_sentiment(p) for p in str_predictions]
        correct += (np.array(sent_preds) == np.array(batch["label"])).sum()
    return round((correct / len(ds)) * 100, 1)

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    llm_name = "Qwen/Qwen3-4B"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    print("GSM8K:", eval_gsm8k(llm, tokenizer, num_samples=100))
    print("BoolQ:", eval_boolq(llm, tokenizer, num_samples=100))
    print("SST2:", eval_sst2(llm, tokenizer, num_samples=100))

if __name__ == "__main__":
    main()
