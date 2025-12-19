import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from loguru import logger
from transformers import GenerationConfig


def _extract_answer(text: str) -> float | None:
    if "<start_of_turn>model" in text:
        text = text.split("<start_of_turn>model")[-1]
    text = text.replace(",", "")
    for pat in (r"####\s*(-?\d+(?:\.\d+)?)", r"answer is\s*(-?\d+(?:\.\d+)?)"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(nums[-1]) if nums else None


def setup_generation_config(model):
    gc = model.generation_config
    model.generation_config = GenerationConfig(
        bos_token_id=gc.bos_token_id,
        eos_token_id=gc.eos_token_id,
        pad_token_id=gc.pad_token_id,
        do_sample=False,
        use_cache=True,
        cache_implementation=None,
    )


def _get_equally_spaced_indices(total_size: int, num_samples: int) -> list[int]:
    if num_samples >= total_size:
        return list(range(total_size))
    step = total_size / num_samples
    return [int(i * step) for i in range(num_samples)]


def log_gsm8k_table(a: dict, b: dict, a_name: str = "teacher", b_name: str = "student"):
    lines = [
        f"{'model':<10} {'GSM8K':>6} {'correct':>7} {'total':>5}",
        f"{a_name:<10} {a['accuracy']*100:>6.1f} {a['correct']:>7} {a['total']:>5}",
        f"{b_name:<10} {b['accuracy']*100:>6.1f} {b['correct']:>7} {b['total']:>5}",
    ]
    logger.info("\n" + "\n".join(lines))


@torch.no_grad()
def eval_gsm8k(model, tokenizer, device="cuda", num_samples=100, batch_size=32):
    model.eval()
    
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    original_pad_token = tokenizer.pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    full_data = load_dataset("gsm8k", "main", split="test")
    total_size = len(full_data)
    
    indices = _get_equally_spaced_indices(total_size, num_samples)
    data = full_data.select(indices)
    
    correct = 0
    total = len(data)
    
    prompts = []
    truths = []
    
    for row in data:
        question = row['question'] + "\nLet's think step by step. At the end, output the final answer after '####'."
        
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
        
        truth_str = row['answer'].split("#### ")[-1].strip().replace(',', '')
        truths.append(float(truth_str))

    for i in tqdm(range(0, total, batch_size), desc="Eval GSM8K", leave=False):
        batch_prompts = prompts[i:i + batch_size]
        batch_truths = truths[i:i + batch_size]
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024 
        ).to(device)
        
        input_len = inputs.input_ids.shape[1]
        
        out = model.generate(
            **inputs,
            max_new_tokens=512,
        )
        
        generated_tokens = out[:, input_len:]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for text, truth in zip(decoded_outputs, batch_truths):
            pred = _extract_answer(text)
            if pred is not None and abs(pred - truth) < 1e-4:
                correct += 1

    tokenizer.padding_side = original_padding_side
    if original_pad_token is not None:
        tokenizer.pad_token = original_pad_token

    accuracy = correct / total
    res = {"accuracy": float(accuracy), "correct": int(correct), "total": int(total)}
    logger.info(f"[GSM8K] Final: {correct}/{total} correct = {accuracy*100:.1f}%")
    return res
