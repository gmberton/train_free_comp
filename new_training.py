
import torch
import wandb
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_commons import make_deterministic, initialize_logger

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--llm_name', type=str, default="google/gemma-3-4b-it", help='_')
# parser.add_argument('--extended_tokenizer_path', type=str, default="./data/extended_tokenizer_google-gemma-3-270m-it_100tokens", help='_')
parser.add_argument('--llm_name', type=str, default="Qwen/Qwen3-0.6B", help='_')
parser.add_argument('--extended_tokenizer_path', type=str, default="./data/extended_tokenizer_Qwen-Qwen3-4B_100tokens", help='_')
parser.add_argument('-nq', '--num_questions', type=int, default=100000, help='max 659808')
parser.add_argument('-bs', '--batch_size', type=int, default=4, help='_')
parser.add_argument('-ne', '--num_epochs', type=int, default=10, help='_')
parser.add_argument('-ipe', '--num_iters_per_epoch', type=int, default=1000, help='_')
parser.add_argument('--lr', type=float, default=0.1, help='_')
parser.add_argument('--device', type=str, default="cuda", help='_')
parser.add_argument('--max_combined_chars', type=int, default=3000, help='Q&A longer than this will be filtered out')
parser.add_argument("--exp_name", type=str, default="default", help="_")
parser.add_argument("--nowb", action="store_true", help="no wandb logging")
args = parser.parse_args()
args.num_questions = min(args.num_questions, 659808)

log_dir = initialize_logger(exp_name=args.exp_name, stdout='INFO')
logger.info(f"All arguments: {args}")
make_deterministic(2)
if args.nowb:
    wandb.log = lambda _: None
else:
    run = wandb.init(project=f"tfc3", name=args.exp_name)
    logger.info(f"wandb: ⭐️ View project at {run.get_project_url()}")

#### Create new and old tokenizers
old_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
new_tokenizer = AutoTokenizer.from_pretrained(args.extended_tokenizer_path, trust_remote_code=True)
# Sanity check: ensure new tokenizer is an extension of old tokenizer
assert len(new_tokenizer) >= len(old_tokenizer)
assert all(old_tokenizer.convert_ids_to_tokens(i) == new_tokenizer.convert_ids_to_tokens(i) for i in range(len(old_tokenizer)))
new_token_ids = list(range(len(old_tokenizer), len(new_tokenizer)))
assert len(new_token_ids) == 100

#### Create data pairs
dataset_infinity_instruct = load_dataset("BAAI/Infinity-Instruct", "0625", trust_remote_code=True)
all_pairs = []
for sample in tqdm(dataset_infinity_instruct['train'].select(random.sample(range(659808), args.num_questions)), desc="Creating Q-A pairs", ncols=100):
    question, answer = sample['conversations'][:2]
    question, answer = question['value'], answer['value']
    if len(question) + len(answer) < args.max_combined_chars:
        all_pairs.append((question, answer))

tmp = len(all_pairs)
all_pairs = [(q, a) for q, a in tqdm(all_pairs, desc="Filtering", ncols=100) if len(set(new_token_ids).intersection(set(new_tokenizer.encode(q)))) != 0]
logger.info(f"Out of {args.num_questions} Q&A pairs, of which {tmp} were selected, keeping only the {len(all_pairs)} that contain new tokens")

# Split train and valid data
random.shuffle(all_pairs)
split_idx = int(len(all_pairs) * 0.8)  # 80% of data is training, rest is validation
train_all_pairs, valid_all_pairs = all_pairs[:split_idx], all_pairs[split_idx:]
logger.info(f"Train pairs: {len(train_all_pairs)}, Val pairs: {len(valid_all_pairs)}")

#### Create model. Untie weights. Expand embedding table.
llm = AutoModelForCausalLM.from_pretrained(args.llm_name, trust_remote_code=True, torch_dtype=torch.float32).to(args.device)
input_emb = llm.get_input_embeddings()
model_vocab_size, hidden_dim = input_emb.weight.shape
output_emb = llm.get_output_embeddings() if hasattr(llm, 'get_output_embeddings') else None
is_tied = (hasattr(llm.config, 'tie_word_embeddings') and llm.config.tie_word_embeddings) or \
          (output_emb and input_emb.weight.data_ptr() == output_emb.weight.data_ptr())
if is_tied:
    lm_head = llm.lm_head
    new_lm_head = nn.Linear(hidden_dim, model_vocab_size, bias=lm_head.bias is not None).to(args.device)
    new_lm_head.weight.data.copy_(lm_head.weight.data[:model_vocab_size])
    if lm_head.bias is not None:
        new_lm_head.bias.data.copy_(lm_head.bias.data[:model_vocab_size])
    llm.lm_head = new_lm_head
    if hasattr(llm.config, 'tie_word_embeddings'):
        llm.config.tie_word_embeddings = False

# Resize input embeddings and create learnable embeddings
llm.resize_token_embeddings(len(new_tokenizer))
learnable_embeddings = {tkn_id: nn.Parameter(torch.randn(hidden_dim, device=args.device)) for tkn_id in new_token_ids}
optimizer = torch.optim.Adam(learnable_embeddings.values(), lr=args.lr)
_ = llm.requires_grad_(False)
_ = llm.eval()
def sync_embeddings():  # Set learnable_embeddings into embedding table
    with torch.no_grad():
        for token_id in new_token_ids:
            llm.get_input_embeddings().weight.data[token_id] = learnable_embeddings[token_id].data

sync_embeddings()

def compute_loss(batch_pairs, do_backward):
    losses = []
    for q, a in batch_pairs:
        qq_chat = old_tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, enable_thinking=False)
        qa_chat = old_tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False, enable_thinking=False)
        qq_new_tkns = new_tokenizer.encode(qq_chat)
        qq_old_tkns = old_tokenizer.encode(qq_chat)
        qa_old_tkns = old_tokenizer.encode(qa_chat)
        assert isinstance(qq_new_tkns, list)
        assert qa_old_tkns[:len(qq_old_tkns)] == qq_old_tkns
        assert len(qa_old_tkns) > len(qq_old_tkns)
        aa_old_tkns = qa_old_tkns[len(qq_old_tkns):]
        # Compute teacher logits
        with torch.no_grad():
            teacher_inputs_embeds = llm.get_input_embeddings()(torch.tensor(qa_old_tkns, device=args.device).reshape(1, -1))
            teacher_logits = llm(inputs_embeds=teacher_inputs_embeds).logits
            teacher_logits = teacher_logits[0, -len(aa_old_tkns):]  # Select only logits of answer
        # Compute student logits
        qq_new_aa_old_tkns = qq_new_tkns + aa_old_tkns
        if qa_old_tkns == qq_new_aa_old_tkns:
            # print(f"No new tokens here, continue")
            raise RuntimeError("This should never happen because I've pre-filtered Q&A pairs")  # TODO remove
        else:
            if not len(qa_old_tkns) > len(qq_new_aa_old_tkns):
                import ipdb; ipdb.set_trace()
            assert len(qa_old_tkns) > len(qq_new_aa_old_tkns)
            # print(f"Found {len(qa_old_tkns) - len(qq_new_aa_old_tkns)} new tokens")
        student_inputs_embeds = llm.get_input_embeddings()(torch.tensor(qq_new_aa_old_tkns, device=args.device).reshape(1, -1))
        for idx, tkn in enumerate(qq_new_aa_old_tkns):
            if tkn in new_token_ids:
                student_inputs_embeds[0, idx] = learnable_embeddings[tkn]
        student_logits = llm(inputs_embeds=student_inputs_embeds).logits
        student_logits = student_logits[0, -len(aa_old_tkns):]  # Select only logits of answer
        loss = F.mse_loss(student_logits, teacher_logits)
        if do_backward:
            loss.backward()
        losses.append(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    return np.mean(losses)

for num_epoch in range(args.num_epochs):
    tqdm_bar = tqdm(range(args.num_iters_per_epoch), ncols=100)
    train_losses, valid_losses = [], []
    for num_iter in tqdm_bar:
        train_batch_pairs = random.choices(train_all_pairs, k=args.batch_size)
        train_loss = compute_loss(train_batch_pairs, do_backward=True)
        with torch.no_grad():
            valid_batch_pairs = random.choices(valid_all_pairs, k=args.batch_size)
            valid_loss = compute_loss(valid_batch_pairs, do_backward=False)
        wandb.log({"train_losses": train_loss, "valid_losses": valid_loss})
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        tqdm_bar.desc = f"train_losses = {np.mean(train_losses):.3f}, valid_loss = {np.mean(valid_losses):.3f}"
    wandb.log({"train_loss": np.mean(train_losses), "valid_loss": np.mean(valid_losses)})
    logger.info(f"train_losses = {np.mean(train_losses):.3f}, valid_loss = {np.mean(valid_losses):.3f}")
