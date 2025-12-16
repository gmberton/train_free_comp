"""
Train learnable embeddings for new tokens added to an extended tokenizer.
Uses knowledge distillation: old tokenizer (teacher) vs new tokenizer (student).
"""

import torch
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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--llm_name', type=str, default="google/gemma-3-270m-it", help='_')
parser.add_argument('--extended_tokenizer_path', type=str, default="./data/extended_tokenizer_google-gemma-3-270m-it_100tokens", help='_')
parser.add_argument('--num_questions', type=int, default=65, help='max 659808')
parser.add_argument('--train_split', type=float, default=0.8, help='_')
parser.add_argument('--batch_size', type=int, default=4, help='_')
parser.add_argument('--grad_accum_steps', type=int, default=2, help='_')
parser.add_argument('--num_epochs', type=int, default=10, help='_')
parser.add_argument('--learning_rate', type=float, default=0.1, help='_')
parser.add_argument('--device', type=str, default="cuda", help='_')
parser.add_argument('--max_combined_chars', type=int, default=3000, help='_')
parser.add_argument("--exp_name", type=str, default="default", help="_")
parser.add_argument("--nowb", action="store_true", help="no wandb logging")
args = parser.parse_args()

log_dir = initialize_logger(exp_name="default", stdout='INFO')

random.seed(0)
torch.manual_seed(0)

if args.nowb:
    wandb.log = lambda _: None
else:
    wandb.init(project=f"my_compressor_{exp_prefix}", name=args.exp_name)


# Load tokenizers
logger.info("Loading tokenizers...")
old_tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
new_tokenizer = AutoTokenizer.from_pretrained(args.extended_tokenizer_path, trust_remote_code=True)
old_vocab_size = len(old_tokenizer)
new_vocab_size = len(new_tokenizer)
# Sanity check: ensure new tokenizer is an extension of old tokenizer
assert new_vocab_size >= old_vocab_size
assert all(old_tokenizer.convert_ids_to_tokens(i) == new_tokenizer.convert_ids_to_tokens(i) for i in range(old_vocab_size))
new_token_ids = list(range(old_vocab_size, new_vocab_size))
logger.info(f"New tokens: {len(new_token_ids)} ({old_vocab_size} -> {new_vocab_size})")

# Load and setup model
logger.info("Loading model...")
llm = AutoModelForCausalLM.from_pretrained(args.llm_name, trust_remote_code=True, torch_dtype=torch.float32).to(args.device)
input_emb = llm.get_input_embeddings()
hidden_dim = input_emb.weight.shape[1]
model_vocab_size = input_emb.weight.shape[0]

# De-tie embeddings if needed
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
    logger.success("Input and output embeddings are now separate")

# Resize input embeddings and create learnable embeddings
llm.resize_token_embeddings(new_vocab_size)
learnable_embeddings = nn.Parameter(torch.randn(len(new_token_ids), hidden_dim, device=args.device))
optimizer = torch.optim.Adam([learnable_embeddings], lr=args.learning_rate)
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
pairs = []
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
            pairs.append((user_content, content))
            user_content = None

pairs = list(dict.fromkeys(pairs))
if len(pairs) == 0:
    logger.error("No Q-A pairs found in dataset!")
    raise ValueError("No Q-A pairs found in dataset!")

# Filter by combined character length (Q + A)
initial_count = len(pairs)
pairs = [(q, a) for q, a in pairs if len(q) + len(a) <= args.max_combined_chars]
filtered_count = initial_count - len(pairs)
if filtered_count > 0:
    logger.info(f"Filtered out {filtered_count} pairs (Q+A > {args.max_combined_chars} chars)")
logger.info(f"Remaining pairs: {len(pairs)}")

random.shuffle(pairs)
split_idx = int(len(pairs) * args.train_split)
train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]
logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

def get_batch_logits(batch_pairs, llm, tokenizer, learnable_embeddings=None, new_token_ids=None, old_tokenizer=None):
    """Extract answer logits for a batch. If learnable_embeddings provided, uses new_tokenizer for Q, old_tokenizer for A."""
    if learnable_embeddings is not None:
        # Student: new tokenizer for questions, old tokenizer for answers
        # Get answer tokens from old tokenizer (to match teacher exactly)
        full_texts_old = [old_tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False) for q, a in batch_pairs]
        prompt_texts_old = [old_tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q, _ in batch_pairs]
        prompt_tokens_old = [old_tokenizer.encode(p, add_special_tokens=False) for p in prompt_texts_old]
        full_tokens_old = [old_tokenizer.encode(f, add_special_tokens=False) for f in full_texts_old]
        answer_tokens = [full_tok[len(prompt_tok):] for prompt_tok, full_tok in zip(prompt_tokens_old, full_tokens_old)]
        
        # Tokenize questions with new tokenizer
        prompt_texts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q, _ in batch_pairs]
        prompt_tokens = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_texts]

        new_tokens = [token for tokens in prompt_tokens for token in tokens if token in new_token_ids]
        old_tokens = [token for tokens in prompt_tokens for token in tokens if token not in new_token_ids]
        logger.debug(f"New tokens: {len(new_tokens)}, Old tokens: {len(old_tokens)}")
        if len(new_tokens) == 0:
            return None

        # Combine: new tokenizer prompt + old tokenizer answer
        full_tokens = [p + a for p, a in zip(prompt_tokens, answer_tokens)]
        
        # Pad and create inputs - use old_tokenizer's pad_token_id since answer tokens come from old tokenizer
        pad_token_id = old_tokenizer.pad_token_id if old_tokenizer.pad_token_id is not None else 0
        max_len = max(len(t) for t in full_tokens)
        input_ids = torch.full((len(batch_pairs), max_len), pad_token_id, dtype=torch.long, device=args.device)
        attention_mask = torch.zeros((len(batch_pairs), max_len), dtype=torch.long, device=args.device)
        prompt_lens = []
        answer_start_positions = []  # Track where answer tokens start in each sequence
        
        for i, (prompt_tok, answer_tok, full_tok) in enumerate(zip(prompt_tokens, answer_tokens, full_tokens)):
            prompt_lens.append(len(prompt_tok))
            answer_start_positions.append(len(prompt_tok))  # Answer starts right after prompt
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
        
        # Extract answer tokens and logits - use answer_start_positions to ensure alignment
        # We extract from answer_start_positions-1 to seq_lens-1 (same as pl-1:sl-1 but using answer positions)
        extracted_tokens = []
        extracted_logits = []
        for j, (ans_start, sl) in enumerate(zip(answer_start_positions, seq_lens)):
            # Extract answer tokens: from ans_start-1 (last prompt token predicts first answer token) to sl-1 (last answer token)
            extracted_tokens.append(input_ids[j, ans_start-1:sl-1])
            extracted_logits.append(outputs.logits[j, ans_start-1:sl-1])
        
        return [torch.cat(extracted_tokens), torch.cat(extracted_logits)]
    else:
        # Teacher: old tokenizer for everything
        full_texts = [tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False) for q, a in batch_pairs]
        prompt_texts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q, _ in batch_pairs]
        prompt_tokens = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_texts]
        prompt_lens = [len(pt) for pt in prompt_tokens]
        
        # Tokenize with padding - ensure consistent padding
        inputs = tokenizer(full_texts, padding=True, return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            outputs = llm(**inputs)
        input_ids = inputs.input_ids
        seq_lens = inputs.attention_mask.sum(dim=1)
        
        # Extract answer tokens and logits
        return [
            torch.cat([input_ids     [j, pl-1:sl-1] for j, (pl, sl) in enumerate(zip(prompt_lens, seq_lens))]),
            torch.cat([outputs.logits[j, pl-1:sl-1] for j, (pl, sl) in enumerate(zip(prompt_lens, seq_lens))])
        ]

logger.info("Starting training...")

for epoch in range(args.num_epochs):
    train_losses, val_losses = [], []
    logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
    
    # Train
    tqdm_bar = tqdm(range(0, len(train_pairs), args.batch_size), desc=f"Train Epoch {epoch+1}")
    for i in tqdm_bar:
        batch = train_pairs[i:i+args.batch_size]
        with torch.no_grad():
            teacher_tkns, teacher_logits = get_batch_logits(batch, llm, old_tokenizer)
        student_tkns, student_logits = get_batch_logits(batch, llm, new_tokenizer, learnable_embeddings, new_token_ids, old_tokenizer)
        if student_logits is None:
            continue
        
        # Validation: ensure teacher and student extract the same number of tokens
        if teacher_tkns.shape[0] != student_tkns.shape[0]:
            logger.warning(f"Token count mismatch: teacher={teacher_tkns.shape[0]}, student={student_tkns.shape[0]}. Skipping batch.")
            continue
        
        # Validation: ensure the tokens match (they should, since answers come from old tokenizer)
        if not torch.equal(teacher_tkns, student_tkns):
            logger.warning(f"Token mismatch detected! Teacher and student tokens don't match. This indicates a padding/alignment issue.")
            # Log first few mismatches for debugging
            mismatch_mask = teacher_tkns != student_tkns
            if mismatch_mask.any():
                first_mismatch_idx = mismatch_mask.nonzero(as_tuple=True)[0][0].item()
                logger.debug(f"First mismatch at index {first_mismatch_idx}: teacher={teacher_tkns[first_mismatch_idx].item()}, student={student_tkns[first_mismatch_idx].item()}")
        
        batch_loss = F.mse_loss(student_logits, teacher_logits) / args.grad_accum_steps
        batch_loss.backward()
        if (i // args.batch_size + 1) % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            sync_embeddings()
        train_losses.append(batch_loss.item() * args.grad_accum_steps)
        tqdm_bar.desc = f"batch_loss={batch_loss.item() * args.grad_accum_steps:.2f}"
    if len(train_pairs) % (args.batch_size * args.grad_accum_steps) != 0:
        wandb.log({"train_loss": batch_loss.item() * args.grad_accum_steps})
        optimizer.step()
        optimizer.zero_grad()
        sync_embeddings()
    
    # Validate
    with torch.no_grad():
        for i in tqdm(range(0, len(val_pairs), args.batch_size), desc=f"Val Epoch {epoch+1}"):
            batch = val_pairs[i:i+args.batch_size]
            teacher_tkns, teacher_logits = get_batch_logits(batch, llm, old_tokenizer)
            student_tkns, student_logits = get_batch_logits(batch, llm, new_tokenizer, learnable_embeddings, new_token_ids, old_tokenizer)
            if student_logits is None:
                continue
            batch_loss = F.mse_loss(student_logits, teacher_logits)
            val_losses.append(batch_loss.item())
            wandb.log({"val_loss": batch_loss.item()})
    
    avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
    logger.info(f"Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    wandb.log({"train_losses": avg_train_loss, "val_losses": avg_val_loss})

logger.success("Training complete!")
