# Hyperparameters
TOKENIZER_NAME = "google/gemma-3-270m-it"
NUM_TOKENS_TO_GENERATE = 10000
NUM_QUESTIONS = 65980  # max 659808

import random
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer
from loguru import logger
import os
import sys

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}")

random.seed(0)
os.makedirs("./data", exist_ok=True)

# Load and extract questions
from datasets import load_dataset
dataset = load_dataset("BAAI/Infinity-Instruct", "0625", trust_remote_code=True)
questions = []
for example in tqdm(dataset['train'].select(range(NUM_QUESTIONS)), desc="Extracting questions"):
    if 'conversations' in example and isinstance(example['conversations'], list):
        for conv in example['conversations']:
            if isinstance(conv, dict):
                content = (conv.get('value') if conv.get('from') in ['user', 'human'] 
                          else conv.get('content') if conv.get('role') == 'user' 
                          else conv.get('content') if 'content' in conv and conv.get('role') != 'assistant' 
                          else '').strip()
                if content:
                    questions.append(content)

# Remove duplicates
questions = list(dict.fromkeys(q for q in questions if q))
logger.info(f"Extracted {len(questions)} unique questions")

logger.info(f"Total questions available: {len(questions)}")
questions = questions[:NUM_QUESTIONS]
logger.info(f"Using {len(questions)} questions for processing")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
logger.success(f"Tokenizer loaded: {TOKENIZER_NAME}")

# Tokenize
def parse(contexts):
    contexts = list(set(contexts))
    random.shuffle(contexts)
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
for _ in tqdm(range(NUM_TOKENS_TO_GENERATE), desc="Finding token pairs"):
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
model_name_safe = TOKENIZER_NAME.replace("/", "-")
tokenizer_path = f"./data/extended_tokenizer_{model_name_safe}_{NUM_TOKENS_TO_GENERATE}tokens"
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
logger.info(f"Decrease with {NUM_TOKENS_TO_GENERATE:>5} new tokens: {decrease:,} tokens ({decrease_percent:.2f}%)")
logger.info("="*80)
