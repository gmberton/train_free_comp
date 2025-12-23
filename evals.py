
import re
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
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

def get_choice_abcd(text):
    # Parse A/B/C/D after ####, otherwise grab the last one found
    s = text.strip().upper()
    m = re.findall(r"####\s*([ABCD])\b", s)
    if not m:
        m = re.findall(r"\b([ABCD])\b", s)
    return m[-1] if m else None

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
    num_tkns = int(inputs.attention_mask.sum())
    with torch.amp.autocast(device_type=model.device.type, dtype=model.dtype):
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, top_p=None, top_k=None, temperature=None)
    # Slice output to remove input tokens
    answer_tokens = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer_tokens, num_tkns


@torch.no_grad()
def eval_gsm8k(model, tokenizer, num_samples=100, bsz=32):
    _prep_tokenizer(tokenizer)
    ds = load_dataset("gsm8k", "main", split="test")
    ds = _subsample_evenly(ds, num_samples)
    correct = 0
    tot_num_tkns = 0
    for i in tqdm(range(0, len(ds), bsz), desc="Eval GSM8K", ncols=100, leave=False):
        batch = ds[i:i+bsz]
        questions = [q + "\nBe brief and output final answer after '####'." for q in batch['question']]
        str_predictions, num_tkns = compute_preds(model, tokenizer, questions)
        tot_num_tkns += num_tkns
        num_predictions = [get_num(pred) for pred in str_predictions]
        ground_truths = [int(d.split('#### ')[1].replace(',', '')) for d in batch['answer']]
        correct += (np.array(num_predictions) == np.array(ground_truths)).sum()
    acc = correct / len(ds)
    return round(acc * 100, 1), tot_num_tkns // num_samples

@torch.no_grad()
def eval_boolq(model, tokenizer, num_samples=100, bsz=32):
    _prep_tokenizer(tokenizer)
    ds = load_dataset("super_glue", "boolq", split="validation")
    ds = _subsample_evenly(ds, num_samples)
    correct = 0
    tot_num_tkns = 0
    for i in tqdm(range(0, len(ds), bsz), desc="Eval BoolQ", ncols=100, leave=False):
        batch = ds[i:i+bsz]
        questions = [
            f"Passage:\n{p}\n\nQuestion: {q}\nAnswer True or False. Be brief and output final answer after '####'."
            for p, q in zip(batch["passage"], batch["question"])
        ]
        str_predictions, num_tkns = compute_preds(model, tokenizer, questions)
        tot_num_tkns += num_tkns
        bool_preds = [get_bool(p) for p in str_predictions]
        correct += (np.array(bool_preds) == np.array(batch["label"])).sum()
    return round((correct / len(ds)) * 100, 1), tot_num_tkns // num_samples

@torch.no_grad()
def eval_sst2(model, tokenizer, num_samples=100, bsz=32):
    _prep_tokenizer(tokenizer)
    ds = load_dataset("glue", "sst2", split="validation")
    ds = _subsample_evenly(ds, num_samples)
    correct = 0
    tot_num_tkns = 0
    for i in tqdm(range(0, len(ds), bsz), desc="Eval SST2", ncols=100, leave=False):
        batch = ds[i:i+bsz]
        questions = [
            f"Sentence: {s}\nClassify sentiment as Positive or Negative. Be brief and output final answer after '####'."
            for s in batch["sentence"]
        ]
        str_predictions, num_tkns = compute_preds(model, tokenizer, questions)
        tot_num_tkns += num_tkns
        sent_preds = [get_sentiment(p) for p in str_predictions]
        correct += (np.array(sent_preds) == np.array(batch["label"])).sum()
    return round((correct / len(ds)) * 100, 1), tot_num_tkns // num_samples

@torch.no_grad()
def eval_hellaswag(model, tokenizer, num_samples=100, bsz=16):
    _prep_tokenizer(tokenizer)
    ds = load_dataset("hellaswag", split="validation")
    ds = _subsample_evenly(ds, num_samples)
    correct = 0
    tot_num_tkns = 0
    letters = ["A", "B", "C", "D"]
    for i in tqdm(range(0, len(ds), bsz), desc="Eval HellaSwag", ncols=100, leave=False):
        batch = ds[i:i+bsz]
        questions = []
        for ctx, endings in zip(batch["ctx"], batch["endings"]):
            opts = "\n".join([f"{letters[j]}) {endings[j]}" for j in range(4)])
            questions.append(f"Context: {ctx}\n\nOptions:\n{opts}\n\nChoose A, B, C, or D. Be brief and output final answer after '####'.")
        str_predictions, num_tkns = compute_preds(model, tokenizer, questions)
        tot_num_tkns += num_tkns
        preds = [get_choice_abcd(p) for p in str_predictions]
        gts = [letters[int(x)] for x in batch["label"]]
        correct += (np.array(preds) == np.array(gts)).sum()
    return round((correct / len(ds)) * 100, 1), tot_num_tkns // num_samples

@torch.no_grad()
def eval_mmlu_subject(model, tokenizer, subject, num_samples=100, bsz=8):
    _prep_tokenizer(tokenizer)
    letters = ["A", "B", "C", "D"]
    correct = 0
    tot_num_tkns = 0
    ds = load_dataset("cais/mmlu", subject, split="test")
    ds = _subsample_evenly(ds, num_samples)
    for i in tqdm(range(0, len(ds), bsz), desc=f"Eval MMLU:{subject}", ncols=100, leave=False):
        batch = ds[i:i+bsz]
        questions = []
        for q, choices in zip(batch["question"], batch["choices"]):
            opts = "\n".join([f"{letters[j]}) {choices[j]}" for j in range(4)])
            questions.append(
                f"Question: {q}\n\nOptions:\n{opts}\n\nChoose A, B, C, or D. Be brief and output final answer after '####'."
            )
        str_predictions, num_tkns = compute_preds(model, tokenizer, questions)
        tot_num_tkns += num_tkns
        preds = [get_choice_abcd(p) for p in str_predictions]
        gts = [letters[int(x)] for x in batch["answer"]]
        correct += (np.array(preds) == np.array(gts)).sum()
    return round((correct / max(1, len(ds))) * 100, 1), tot_num_tkns // max(1, len(ds))

@torch.no_grad()
def compute_evals(llm, tokenizer, num_samples=100, bsz=32):
    gsm_acc, gsm_tkns_per_qst = eval_gsm8k(llm, tokenizer, num_samples=num_samples, bsz=bsz)
    boo_acc, boo_tkns_per_qst = eval_boolq(llm, tokenizer, num_samples=num_samples, bsz=bsz)
    sst_acc, sst_tkns_per_qst = eval_sst2(llm, tokenizer, num_samples=num_samples, bsz=bsz)
    hel_acc, hel_tkns_per_qst = eval_hellaswag(llm, tokenizer, num_samples=num_samples, bsz=bsz)
    mmlu_cs_acc, mmlu_cs_tkns_per_qst = eval_mmlu_subject(llm, tokenizer, "college_computer_science", num_samples=num_samples, bsz=bsz)
    mmlu_sec_acc, mmlu_sec_tkns_per_qst = eval_mmlu_subject(llm, tokenizer, "computer_security", num_samples=num_samples, bsz=bsz)
    mmlu_psych_acc, mmlu_psych_tkns_per_qst = eval_mmlu_subject(llm, tokenizer, "high_school_psychology", num_samples=num_samples, bsz=bsz)
    mmlu_philo_acc, mmlu_philo_tkns_per_qst = eval_mmlu_subject(llm, tokenizer, "philosophy", num_samples=num_samples, bsz=bsz)
    logger.debug(
        f"gsm_tkns_per_qst={gsm_tkns_per_qst}, boo_tkns_per_qst={boo_tkns_per_qst}, sst_tkns_per_qst={sst_tkns_per_qst}, "
        f"hel_tkns_per_qst={hel_tkns_per_qst}, mmlu_cs_tkns_per_qst={mmlu_cs_tkns_per_qst}, "
        f"mmlu_sec_tkns_per_qst={mmlu_sec_tkns_per_qst}, mmlu_psych_tkns_per_qst={mmlu_psych_tkns_per_qst}, "
        f"mmlu_philo_tkns_per_qst={mmlu_philo_tkns_per_qst}"
    )
    tkns_per_qst = (gsm_tkns_per_qst + boo_tkns_per_qst + sst_tkns_per_qst + hel_tkns_per_qst + mmlu_cs_tkns_per_qst + \
                    mmlu_sec_tkns_per_qst + mmlu_psych_tkns_per_qst + mmlu_philo_tkns_per_qst) // 8
    return {
        "GSM8k": gsm_acc, "BoolQ": boo_acc, "SST2": sst_acc, "HellaSwag": hel_acc, "MMLU_CS": mmlu_cs_acc,
        "MMLU_Sec": mmlu_sec_acc, "MMLU_Psych": mmlu_psych_acc, "MMLU_Philo": mmlu_philo_acc
    }, tkns_per_qst

def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B", help="_")
    p.add_argument("--device", default="cuda", help="_")
    p.add_argument("--num_samples", type=int, default=100, help="_")
    p.add_argument("--bsz", type=int, default=32, help="_")
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device)
    logger.info("Starting evals")
    logger.info("GSM8K:" + str(eval_gsm8k(llm, tokenizer, num_samples=args.num_samples, bsz=args.bsz)))
    logger.info("BoolQ:" + str(eval_boolq(llm, tokenizer, num_samples=args.num_samples, bsz=args.bsz)))
    logger.info("SST2:" + str(eval_sst2(llm, tokenizer, num_samples=args.num_samples, bsz=args.bsz)))
    logger.info("HellaSwag:" + str(eval_hellaswag(llm, tokenizer, num_samples=args.num_samples, bsz=args.bsz)))
    logger.info("MMLU_CS:" + str(eval_mmlu_subject(llm, tokenizer, "college_computer_science", num_samples=args.num_samples, bsz=args.bsz//4)))
    logger.info("MMLU_Sec:" + str(eval_mmlu_subject(llm, tokenizer, "computer_security", num_samples=args.num_samples, bsz=args.bsz//4)))
    logger.info("MMLU_Psych:" + str(eval_mmlu_subject(llm, tokenizer, "high_school_psychology", num_samples=args.num_samples, bsz=args.bsz//4)))
    logger.info("MMLU_Philo:" + str(eval_mmlu_subject(llm, tokenizer, "philosophy", num_samples=args.num_samples, bsz=args.bsz//4)))
    results, tkns_per_qst = compute_evals(llm, tokenizer, num_samples=args.num_samples, bsz=args.bsz)
    for dataset_name, accuracy in results.items():
        logger.info(f"{dataset_name:<12} accuracy={accuracy}")

if __name__ == "__main__":
    main()
