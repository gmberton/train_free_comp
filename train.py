"""
Train learnable embeddings for new tokens added to an extended tokenizer.
Uses knowledge distillation: old tokenizer (teacher) vs new tokenizer (student).
"""

# TODO Double check the train function
# TODO try to overfit to some val or even train samples
# TODO filter out train / val without new tokens offline
# TODO add KL
# TODO run eval at the beginning

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
from tqdm import tqdm
from loguru import logger

from utils_commons import initialize_logger
import wandb
import argparse
import eval
os.environ["WANDB_SILENT"] = "true"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--llm_name', type=str, default="google/gemma-3-4b-it", help='_')
# parser.add_argument('--extended_tokenizer_path', type=str, default="./data/extended_tokenizer_google-gemma-3-270m-it_100tokens", help='_')
parser.add_argument('--llm_name', type=str, default="Qwen/Qwen3-0.6B", help='_')
parser.add_argument('--extended_tokenizer_path', type=str, default="./data/extended_tokenizer_Qwen-Qwen3-4B_100tokens", help='_')
parser.add_argument('-nq', '--num_questions', type=int, default=100000, help='max 659808')
parser.add_argument('-bs', '--batch_size', type=int, default=4, help='_')
parser.add_argument('-ne', '--num_epochs', type=int, default=10, help='_')
parser.add_argument('-ipe', '--iterations_per_epoch', type=int, default=1000, help='_')
parser.add_argument('--lr', type=float, default=0.1, help='_')
parser.add_argument('--device', type=str, default="cuda", help='_')
parser.add_argument('--max_combined_chars', type=int, default=3000, help='Q&A longer than this will be filtered out')
parser.add_argument("--exp_name", type=str, default="default", help="_")
parser.add_argument("--nowb", action="store_true", help="no wandb logging")
args = parser.parse_args()
args.num_questions = min(args.num_questions, 659808)

log_dir = initialize_logger(exp_name="default", stdout='INFO')
logger.info(f"All arguments: {args}")

random.seed(0)
torch.manual_seed(0)

if args.nowb:
    wandb.log = lambda _: None
else:
    run = wandb.init(project=f"tfc2", name=args.exp_name)
    logger.info(f"wandb: ⭐️ View project at {run.get_project_url()}")

# Load tokenizers
logger.info("Loading tokenizers...")
old_tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
new_tokenizer = AutoTokenizer.from_pretrained(args.extended_tokenizer_path, trust_remote_code=True)

old_tokenizer.padding_side = "right"
new_tokenizer.padding_side = "right"
if old_tokenizer.pad_token is None: old_tokenizer.pad_token = old_tokenizer.eos_token

if new_tokenizer.pad_token is None: new_tokenizer.pad_token = new_tokenizer.eos_token

# Sanity check: ensure new tokenizer is an extension of old tokenizer
assert len(new_tokenizer) >= len(old_tokenizer)
assert all(old_tokenizer.convert_ids_to_tokens(i) == new_tokenizer.convert_ids_to_tokens(i) for i in range(len(old_tokenizer)))
new_token_ids = list(range(len(old_tokenizer), len(new_tokenizer)))
logger.info(f"New tokens: {len(new_token_ids)} ({len(old_tokenizer)} -> {len(new_tokenizer)})")

# Load and setup model
logger.info("Loading model...")
llm = AutoModelForCausalLM.from_pretrained(args.llm_name, trust_remote_code=True, torch_dtype=torch.float32).to(args.device)
input_emb = llm.get_input_embeddings()
model_vocab_size, hidden_dim = input_emb.weight.shape

output_emb = llm.get_output_embeddings() if hasattr(llm, 'get_output_embeddings') else None
is_tied = (hasattr(llm.config, 'tie_word_embeddings') and llm.config.tie_word_embeddings) or \
          (output_emb and input_emb.weight.data_ptr() == output_emb.weight.data_ptr())

if is_tied:
    logger.info("De-tying embeddings...")
    lm_head = llm.lm_head
    new_lm_head = nn.Linear(hidden_dim, model_vocab_size, bias=lm_head.bias is not None).to(args.device)
    new_lm_head.weight.data.copy_(lm_head.weight.data[:model_vocab_size])
    if lm_head.bias is not None:
        new_lm_head.bias.data.copy_(lm_head.bias.data[:model_vocab_size])
    llm.lm_head = new_lm_head
    if hasattr(llm.config, 'tie_word_embeddings'):
        llm.config.tie_word_embeddings = False
    logger.info("Input and output embeddings are now separate")

# Resize input embeddings and create learnable embeddings
llm.resize_token_embeddings(len(new_tokenizer))
learnable_embeddings = nn.Parameter(torch.randn(len(new_token_ids), hidden_dim, device=args.device))
optimizer = torch.optim.Adam([learnable_embeddings], lr=args.lr)
logger.info(f"Created learnable embeddings for {len(new_token_ids)} new tokens")

# Initialize and freeze
def sync_embeddings():
    with torch.no_grad():
        for i, token_id in enumerate(new_token_ids):
            llm.get_input_embeddings().weight.data[token_id] = learnable_embeddings.data[i]

sync_embeddings()
llm.requires_grad_(False)
llm.eval()  # Never use train() mode

# Load data
logger.info("Loading dataset...")
dataset = load_dataset("BAAI/Infinity-Instruct", "0625", trust_remote_code=True)
all_pairs = []
for example in tqdm(dataset['train'].select(range(args.num_questions)), desc="Creating Q-A pairs"):
    if 'conversations' not in example or not isinstance(example['conversations'], list):
        continue
    user_content = None
    for conv in example['conversations']:
        if not isinstance(conv, dict):
            continue
        role, content = conv.get('from', ''), conv.get('value', '').strip()
        if role == 'human' and content:
            user_content = content
        elif role in ['gpt', 'assistant'] and content and user_content:
            all_pairs.append((user_content, content))
            user_content = None

all_pairs = list(dict.fromkeys(all_pairs))

# Filter by combined character length (Q + A)
initial_count = len(all_pairs)
all_pairs = [(q, a) for q, a in all_pairs if len(q) + len(a) <= args.max_combined_chars]
filtered_count = initial_count - len(all_pairs)
if filtered_count > 0:
    logger.info(f"Filtered out {filtered_count} pairs out of {initial_count} (Q+A > {args.max_combined_chars} chars). Remaining pairs: {len(all_pairs)}")

random.shuffle(all_pairs)
split_idx = int(len(all_pairs) * 0.8)  # 80% of data is training, rest is validation
train_all_pairs, valid_all_pairs = all_pairs[:split_idx], all_pairs[split_idx:]
logger.info(f"Train pairs: {len(train_all_pairs)}, Val pairs: {len(valid_all_pairs)}")


def get_batch_logits(batch_pairs, llm, tokenizer, learnable_embeddings=None, new_token_ids=None, old_tokenizer=None):
    """Extract answer logits for a batch. If learnable_embeddings provided, uses new_tokenizer for Q, old_tokenizer for A."""
    if learnable_embeddings is not None:
        # Student: new tokenizer for questions, old tokenizer for answers
        # Get answer tokens from old tokenizer (to match teacher exactly)
        full_texts_old = [old_tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False, enable_thinking=False) for q, a in batch_pairs]
        prompt_texts_old = [old_tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q, _ in batch_pairs]
        prompt_tokens_old = [old_tokenizer.encode(p, add_special_tokens=False) for p in prompt_texts_old]
        full_tokens_old = [old_tokenizer.encode(f, add_special_tokens=False) for f in full_texts_old]
        answer_tokens = [full_tok[len(prompt_tok):] for prompt_tok, full_tok in zip(prompt_tokens_old, full_tokens_old)]
        
        # Tokenize questions with new tokenizer
        prompt_texts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q, _ in batch_pairs]
        prompt_tokens = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_texts]

        new_tokens = [token for tokens in prompt_tokens for token in tokens if token in new_token_ids]
        old_tokens = [token for tokens in prompt_tokens for token in tokens if token not in new_token_ids]
        logger.debug(f"New tokens: {len(new_tokens)}, Old tokens: {len(old_tokens)}")
        if len(new_tokens) == 0:
            return None, None

        # Combine: new tokenizer prompt + old tokenizer answer
        full_tokens = [p + a for p, a in zip(prompt_tokens, answer_tokens)]
        
        # Pad and create inputs
        max_len = max(len(t) for t in full_tokens)
        input_ids = torch.zeros((len(batch_pairs), max_len), dtype=torch.long, device=args.device)
        attention_mask = torch.zeros((len(batch_pairs), max_len), dtype=torch.long, device=args.device)
        prompt_lens = []
        
        for i, (prompt_tok, full_tok) in enumerate(zip(prompt_tokens, full_tokens)):
            prompt_lens.append(len(prompt_tok))
            input_ids[i, :len(full_tok)] = torch.tensor(full_tok, device=args.device)
            attention_mask[i, :len(full_tok)] = 1
        
        # Replace new tokens with learnable embeddings
        token_id_to_idx = {tid: i for i, tid in enumerate(new_token_ids)}
        input_embeds = llm.get_input_embeddings()(input_ids)
        for token_id in new_token_ids:
            mask = input_ids == token_id
            if mask.any():
                input_embeds[mask] = learnable_embeddings[token_id_to_idx[token_id]]
        
        outputs = llm(inputs_embeds=input_embeds, attention_mask=attention_mask)
        seq_lens = attention_mask.sum(dim=1)
    else:
        # Teacher: old tokenizer for everything
        full_texts = [tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False, enable_thinking=False) for q, a in batch_pairs]
        inputs = tokenizer(full_texts, padding=True, return_tensors="pt").to(args.device)
        prompt_texts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q, _ in batch_pairs]
        prompt_lens = [len(tokenizer.encode(p)) for p in prompt_texts]
        
        with torch.no_grad():
            outputs = llm(**inputs)
        input_ids = inputs.input_ids
        seq_lens = inputs.attention_mask.sum(dim=1)
    
    return [
        torch.cat([input_ids     [j, pl-1:sl-1] for j, (pl, sl) in enumerate(zip(prompt_lens, seq_lens))]),
        torch.cat([outputs.logits[j, pl-1:sl-1] for j, (pl, sl) in enumerate(zip(prompt_lens, seq_lens))])
    ]

def compute_loss(pairs_, llm, new_tokenizer, learnable_embeddings, new_token_ids, old_tokenizer, do_backward):
    batch_pairs = random.choices(pairs_, k=args.batch_size)
    student_tkns, student_logits = get_batch_logits(batch_pairs, llm, new_tokenizer, learnable_embeddings, new_token_ids, old_tokenizer)
    if student_logits is None:
        return -1
    with torch.no_grad():
        teacher_tkns, teacher_logits = get_batch_logits(batch_pairs, llm, old_tokenizer)
    assert (student_tkns == teacher_tkns).all()
    loss = F.mse_loss(student_logits, teacher_logits)
    if do_backward:
        str_len = " --- ".join([f"{len(q)}_{len(a)}" for q, a in batch_pairs])
        logger.debug(f"Alloc={torch.cuda.memory_allocated(args.device)/1e9:.2f}G, reserved={torch.cuda.memory_reserved(args.device)/1e9:.2f}G --- {str_len}")
        loss.backward()
    return loss.item()

logger.info("Starting training...")
for num_epoch in range(args.num_epochs):
    train_losses, valid_losses = [], []
    logger.info(f"Epoch {num_epoch+1}/{args.num_epochs}")
    tqdm_bar = tqdm(range(args.iterations_per_epoch), leave=True, ncols=100)
    for num_iter in tqdm_bar:
        train_loss = compute_loss(train_all_pairs, llm, new_tokenizer, learnable_embeddings, new_token_ids, old_tokenizer, do_backward=True)
        with torch.no_grad():
            valid_loss = compute_loss(valid_all_pairs, llm, new_tokenizer, learnable_embeddings, new_token_ids, old_tokenizer, do_backward=False)
        wandb.log({"train_losses": train_loss, "valid_losses": valid_loss})
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        tqdm_bar.desc = f"train_losses = {np.mean(train_losses):.3f}, valid_loss = {np.mean(valid_losses):.3f}"

    wandb.log({"train_loss": np.mean(train_losses), "valid_loss": np.mean(valid_losses)})
    logger.info(f"train_losses = {np.mean(train_losses):.3f}, valid_loss = {np.mean(valid_losses):.3f}")

    teacher_gsm8k = eval.eval_gsm8k(llm, old_tokenizer, device=args.device, num_samples=100)
    student_gsm8k = eval.eval_gsm8k(llm, new_tokenizer, device=args.device, num_samples=100)
    eval.log_gsm8k_table(teacher_gsm8k, student_gsm8k, a_name="teacher", b_name="student")

logger.info("Training complete!")

# eval.setup_generation_config(llm)  # TODO do I need this?

teacher_gsm8k = eval.eval_gsm8k(llm, old_tokenizer, device=args.device, num_samples=1000)
student_gsm8k = eval.eval_gsm8k(llm, new_tokenizer, device=args.device, num_samples=1000)
eval.log_gsm8k_table(teacher_gsm8k, student_gsm8k, a_name="teacher", b_name="student")
