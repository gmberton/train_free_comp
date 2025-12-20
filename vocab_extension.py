
import os
import sys
import random
import argparse
from tqdm import tqdm
from loguru import logger
from collections import Counter
from transformers import AutoTokenizer
from utils_commons import make_deterministic

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--llm_name', type=str, default="Qwen/Qwen3-4B", help='_')
parser.add_argument('-nt', '--num_tokens', type=int, default=100, help='How many new tokens to generate')
args = parser.parse_args()

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}")

make_deterministic(0)
os.makedirs("./data", exist_ok=True)

# Load and extract questions
from datasets import load_dataset

dataset_infinity_instruct = load_dataset("BAAI/Infinity-Instruct", "0625", trust_remote_code=True)
questions = []
for sample in tqdm(dataset_infinity_instruct['train'].select(random.sample(range(659808), args.num_questions)), desc="Creating Q-A pairs", ncols=100):
    question, answer = sample['conversations'][:2]
    questions.append(question['value'])

questions = list(set(questions))
random.shuffle(questions)
logger.info(f"Using {len(questions)} questions for processing")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
logger.success(f"Tokenizer loaded: {args.llm_name}")

# Tokenize
def parse(contexts):
    tokens = []
    for context in tqdm(contexts, desc="Tokenizing"):
        tokens.extend(tokenizer.encode(context, add_special_tokens=False))
    return ''.join(contexts), tokens

train_context, train_tkns = parse(questions)
initial_token_count = len(train_tkns)
logger.info(f"Tokenization complete: {initial_token_count} tokens")

# Merge function
def merge(ids, pair, replacement):
    new_list, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_list.append(replacement)
            i += 2
        else:
            new_list.append(ids[i])
            i += 1
    return new_list

# Find token pairs and extend vocabulary
new_tokens_info = []
for _ in tqdm(range(args.num_tokens), desc="Finding token pairs"):
    pairs = Counter(zip(train_tkns, train_tkns[1:]))
    if len(pairs) == 0:
        logger.error("No token pairs found! Cannot continue.")
        break
    max_count = max(pairs.values())
    best_pair = min([p for p, c in pairs.items() if c == max_count], 
                    key=lambda p: tokenizer.decode(list(p)))
    new_token_str = tokenizer.decode(list(best_pair))
    tokenizer.add_tokens([new_token_str])
    new_id = tokenizer.convert_tokens_to_ids(new_token_str)
    new_tokens_info.append({'id': new_id, 'sub_tokens': list(best_pair), 'text': new_token_str, 'count': max_count})
    train_tkns = merge(train_tkns, best_pair, new_id)

logger.info(f"Found token pairs: {len(train_tkns)} final tokens")

# Print new token information as table
logger.info("\n" + "="*80)
logger.info("New tokens information (first 5):")
logger.info("="*80)
# Table header
logger.info(f"{'Token ID':<12} {'Count':<8} {'Sub-tokens':<25} {'Text':<30}")
logger.info("-" * 80)
# Table rows
for token_info in new_tokens_info[:20]:
    sub_tokens_str = str(token_info['sub_tokens'])[:23] + ('...' if len(str(token_info['sub_tokens'])) > 23 else '')
    text_str = repr(token_info['text'])[:28] + ('...' if len(repr(token_info['text'])) > 28 else '')
    logger.info(f"{token_info['id']:<12} {token_info['count']:<8} {sub_tokens_str:<25} {text_str:<30}")

# Save tokenizer
model_name_safe = args.llm_name.replace("/", "-")
tokenizer_path = f"./data/extended_tokenizer_{model_name_safe}_{args.num_tokens}tokens"
tokenizer.save_pretrained(tokenizer_path)
logger.success(f"Tokenizer saved to '{tokenizer_path}' directory")

# Verify final token count by re-tokenizing with extended tokenizer
verified_token_count = sum(len(tokenizer.encode(q, add_special_tokens=False)) for q in set(questions))

# Print token count summary
final_token_count = len(train_tkns)
decrease = initial_token_count - verified_token_count
decrease_percent = (decrease / initial_token_count * 100) if initial_token_count > 0 else 0

logger.info("\n" + "="*80)
logger.info("Token Count Summary:")
logger.info("="*80)
logger.info(f"Initial tokenizer:              {initial_token_count:,} tokens")
logger.info(f"Final tokenizer:                {verified_token_count:,} tokens")
logger.info(f"Verification (merged count):    {final_token_count:,} tokens (diff: {abs(final_token_count - verified_token_count):,})")
logger.info(f"Decrease with {args.num_tokens:>5} new tokens: {decrease:,} tokens ({decrease_percent:.2f}%)")
logger.info("="*80)
