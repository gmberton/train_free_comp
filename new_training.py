
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

import evals
from utils_commons import make_deterministic, initialize_logger

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--llm_name', type=str, default="google/gemma-3-4b-it", help='_')
# parser.add_argument('--extended_tokenizer_path', type=str, default="./data/extended_tokenizer_google-gemma-3-270m-it_100tokens", help='_')
parser.add_argument('--llm_name', type=str, default="Qwen/Qwen3-4B", help='_')
parser.add_argument('-nt', '--num_tokens', type=int, default=5000, help='How many new tokens to generate')
parser.add_argument('-nq', '--num_questions', type=int, default=659808, help='max 659808')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='_')
parser.add_argument('-ne', '--num_epochs', type=int, default=100, help='_')
parser.add_argument('-ipe', '--num_iters_per_epoch', type=int, default=1000, help='_')
parser.add_argument('--lr', type=float, default=0.1, help='_')
parser.add_argument('--device', type=str, default="cuda", help='_')
parser.add_argument('-mcc', '--max_combined_chars', type=int, default=6000, help='Q&A longer than this will be filtered out')
parser.add_argument("--exp_name", type=str, default="default", help="_")
parser.add_argument("--nowb", action="store_true", help="no wandb logging")
parser.add_argument("-dt", "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"], help="_")
parser.add_argument("-lt", "--loss_type", type=str, default="kl", choices=['mse', 'smoothl1', 'kl', 'codi'], help="_")
args = parser.parse_args()
args.dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
args.num_questions = min(args.num_questions, 659808)

VERSION = 4
if args.exp_name == "default":
    args.exp_name = f"v{VERSION}_nq{args.num_questions}_bs{args.batch_size}"
log_dir = initialize_logger(exp_name=f"v{VERSION}_{args.exp_name}", stdout='INFO')
logger.info(f"All arguments: {args}")
make_deterministic(2)
if args.nowb:
    wandb.log = lambda _: None
else:
    run = wandb.init(project=f"tfc{VERSION}", name=args.exp_name)
    logger.info(f"wandb: ⭐️ View project at {run.get_project_url()}")

#### Create new and old tokenizers
old_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
model_name_safe = args.llm_name.replace("/", "-")
new_tokenizer = AutoTokenizer.from_pretrained(f"./data/extended_tokenizer_{model_name_safe}_{args.num_tokens}tokens", trust_remote_code=True)
# Sanity check: ensure new tokenizer is an extension of old tokenizer
len_old_tokenizer = len(old_tokenizer)
assert len(new_tokenizer) >= len_old_tokenizer
assert all(old_tokenizer.convert_ids_to_tokens(i) == new_tokenizer.convert_ids_to_tokens(i) for i in range(len_old_tokenizer))
new_token_ids = list(range(len_old_tokenizer, len(new_tokenizer)))
assert len(new_token_ids) == args.num_tokens, str(len(new_token_ids))

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
logger.info(f"Out of {args.num_questions} Q&A pairs, of which {tmp} were not too long, keeping only the {len(all_pairs)} that contain new tokens")
# new_pairs = []
# def check_is_subsequence(short, long):
#     # it = iter(long)
#     # return [x for x in short if x not in it]
#     i = 0
#     for x in long:
#         if i < len(short) and x == short[i]:
#             i += 1
#     return short[i:]

# for aa, qq in tqdm(all_pairs, desc="Keep cleanly tokenized tokens", ncols=100):
#     # This loop removes sentences that did not tokenize "cleanly", meaning that do not lead to the creation of extra tokens.
#     # For example the phrase ', then', with old tokenizer is [',', ' then'], with new is []', the', 'n']. It creates 'n', so not clean.
#     old_tkns = old_tokenizer.encode(aa)
#     new_tkns = new_tokenizer.encode(aa)
#     new_old_tkns = [tkn for tkn in new_tkns if tkn not in new_token_ids]
#     if len(check_is_subsequence(new_old_tkns, old_tkns)) == 0:
#         new_pairs.append((aa, qq))

# all_pairs = new_pairs
# logger.info(f"Finally kept {len(all_pairs)} that tokenize cleanly")

# Split train and valid data
random.shuffle(all_pairs)
split_idx = int(len(all_pairs) * 0.9)  # 90% of data is training, rest is validation
train_all_pairs, valid_all_pairs = all_pairs[:split_idx], all_pairs[split_idx:]
logger.info(f"Train pairs: {len(train_all_pairs)}, Val pairs: {len(valid_all_pairs)}")

#### Create model. Untie weights. Expand embedding table.
llm = AutoModelForCausalLM.from_pretrained(args.llm_name, trust_remote_code=True, torch_dtype=args.dtype).to(args.device)
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


def compute_loss(batch_pairs, do_backward):
    losses = []
    assert old_tokenizer.padding_side == "right"
    qq_chats = [old_tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, enable_thinking=False) for q, a in batch_pairs]
    qa_chats = [old_tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False, enable_thinking=False) for q, a in batch_pairs]
    qq_new_tkns = new_tokenizer(qq_chats, return_tensors='pt', padding=True).to(args.device)
    qq_old_tkns = old_tokenizer(qq_chats, return_tensors='pt', padding=True).to(args.device)
    qa_old_tkns = old_tokenizer(qa_chats, return_tensors='pt', padding=True).to(args.device)
    with torch.amp.autocast(device_type=str(llm.device), dtype=llm.dtype):
        with torch.no_grad():
            all_teacher_outputs = llm(**qa_old_tkns, output_hidden_states=args.loss_type=="codi")
            all_teacher_logits = all_teacher_outputs.logits
 
    for num_sample, (q, a) in enumerate(batch_pairs):
        sample_qq_attn_msk = qq_new_tkns.attention_mask[num_sample]
        sample_qq_new_tkns = qq_new_tkns.input_ids[num_sample][:sample_qq_attn_msk.sum()]
        qq_chat = old_tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, enable_thinking=False)
        assert (sample_qq_new_tkns == new_tokenizer(qq_chat, return_tensors='pt').input_ids.to(args.device)).all()
        aa_old_tkns = qa_old_tkns.input_ids[num_sample, qq_old_tkns.attention_mask[num_sample].sum():qa_old_tkns.attention_mask[num_sample].sum()]
        qq_new_aa_old_tkns = torch.cat([sample_qq_new_tkns, aa_old_tkns])
        student_inputs_embeds = llm.get_input_embeddings()(qq_new_aa_old_tkns)
        for num_token in zip(*torch.where(sample_qq_new_tkns >= len_old_tokenizer)):
            tkn_id = int(sample_qq_new_tkns[num_token])
            assert tkn_id in new_token_ids
            assert tkn_id >= len_old_tokenizer
            student_inputs_embeds[num_token] = learnable_embeddings[int(tkn_id)].type(args.dtype)
        with torch.amp.autocast(device_type=str(llm.device), dtype=llm.dtype):
            student_outputs = llm(inputs_embeds=student_inputs_embeds.unsqueeze(0), output_hidden_states=args.loss_type=="codi")
            student_logits = student_outputs.logits
        teacher_logits = all_teacher_logits[num_sample][:qa_old_tkns.attention_mask[num_sample].sum()]  # Remove paddings

        # Remove logits and hidden states of non-answer tokens
        student_logits = student_logits[:, -len(aa_old_tkns):]
        teacher_logits = teacher_logits[-len(aa_old_tkns):].unsqueeze(0)
        print(student_logits.sum())
        print(teacher_logits.sum())

        if args.loss_type == "mse":
            loss = F.mse_loss(student_logits, teacher_logits)
        elif args.loss_type == "smoothl1":
            loss = F.smooth_l1_loss(student_logits, teacher_logits)
        elif args.loss_type == "kl":
            T = 2.0  # temperature > 1.0 softens distributions
            student_probs = torch.nn.functional.log_softmax(student_logits / T, dim=-1)
            teacher_probs = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
            loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (T * T)
        elif args.loss_type == "codi":
            teacher_hidden_states = teacher_outputs.hidden_states
            student_hidden_states = student_outputs.hidden_states
            student_hidden_states = [hstate[:, -len(aa_old_tkns):] for hstate in student_hidden_states]
            teacher_hidden_states = [hstate[:, -len(aa_old_tkns):] for hstate in teacher_hidden_states]
            student_hidden_states = torch.stack(student_hidden_states)
            teacher_hidden_states = torch.stack(teacher_hidden_states)
            std_per_layer = teacher_hidden_states.std([1,2,3]).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            loss = F.smooth_l1_loss(student_hidden_states, teacher_hidden_states, reduction="none")
            loss = (loss / std_per_layer).mean([1,2,3]).sum()
            
        if do_backward:
            loss.backward()
        losses.append(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    exit()
    return np.mean(losses)

for num_epoch in range(args.num_epochs):
    sync_embeddings()
    # old_tkn_results, old_tkns_per_qst = evals.compute_evals(llm, old_tokenizer, num_samples=100)
    # new_tkn_results, new_tkns_per_qst = evals.compute_evals(llm, new_tokenizer, num_samples=100)
    # diffs = {k: old_tkn_results[k] - new_tkn_results[k] for k in new_tkn_results.keys()}
    # if num_epoch == 0:
    #     logger.info(f"{old_tkns_per_qst=}, {new_tkns_per_qst=}")
    # wandb.log(diffs)
    tqdm_bar = tqdm(range(args.num_iters_per_epoch), ncols=100, leave=False)
    train_losses, valid_losses = [], []
    for num_iter in tqdm_bar:
        train_batch_pairs = random.choices(train_all_pairs, k=args.batch_size)
        train_loss = compute_loss(train_batch_pairs, do_backward=True)
        if num_iter % 10 == 0:
            with torch.no_grad():
                valid_batch_pairs = random.choices(valid_all_pairs, k=args.batch_size)
                valid_loss = compute_loss(valid_batch_pairs, do_backward=False)
            wandb.log({"train_losses": train_loss, "valid_losses": valid_loss})
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            tqdm_bar.desc = f"train_losses = {np.mean(train_losses):.3f}, valid_loss = {np.mean(valid_losses):.3f}"
    wandb.log({"train_loss": np.mean(train_losses), "valid_loss": np.mean(valid_losses)})
    logger.info(f"{num_epoch=:>2}, train_losses = {np.mean(train_losses):.3f}, valid_loss = {np.mean(valid_losses):.3f}")

old_tkn_results, old_tkns_per_qst = evals.compute_evals(llm, old_tokenizer, num_samples=1000)
new_tkn_results, new_tkns_per_qst = evals.compute_evals(llm, new_tokenizer, num_samples=1000)
diffs = {k: old_tkn_results[k] - new_tkn_results[k] for k in new_tkn_results.keys()}
logger.info(f"FINAL RESULTS: {old_tkn_results=}")
logger.info(f"FINAL RESULTS: {new_tkn_results=}")
logger.info(f"FINAL RESULTS: {diffs=}")

