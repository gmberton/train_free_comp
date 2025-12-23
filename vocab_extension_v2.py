
import sys
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

from utils_commons import make_deterministic

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--llm_name', type=str, default="Qwen/Qwen3-4B", help='_')
parser.add_argument('-nt', '--num_tokens', type=int, default=10000, help='How many new tokens to generate')
args = parser.parse_args()

model_name_safe = args.llm_name.replace("/", "-")
log_dir = Path(f"./data/extended_tokenizer_{model_name_safe}_{args.num_tokens}tokens")
logger.remove()
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level='INFO')
logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
logger.info(" ".join(sys.argv))

make_deterministic(0)

old_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
new_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)

dataset_infinity_instruct = load_dataset("BAAI/Infinity-Instruct", "0625", trust_remote_code=True)
questions = [sample['conversations'][0]['value'] for sample in tqdm(dataset_infinity_instruct['train'], desc="Creating Q-A pairs", ncols=100)]
# questions = [sample['conversations'][0]['value'] for sample in tqdm(dataset_infinity_instruct['train'].select(list(range(10000))), desc="Creating Q-A pairs", ncols=100)]
questions = list(set(questions))  # Remove duplicates
random.shuffle(questions)
all_text = "".join(questions)
logger.info(f"Using {len(questions)} questions for processing, which are {len(all_text)} characters")

logger.success(f"Start encoding")
tokens = new_tokenizer.encode(all_text)
logger.success(f"Round {round}, finished encoding, found {len(tokens)} tokens")
token_pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
counter = Counter(token_pairs)
counter = sorted(list(counter.items()), key=lambda x: -x[1])
logger.info(f"Found {len(counter)} unique token pairs. The most common is {counter[0][0]} |||{new_tokenizer.decode(counter[0][0])}||| which appears {counter[0][1]} times")
most_common_token_pairs = [new_tokenizer.decode(pair) for pair, _ in counter]
num_added_tokens = 0
# Here I could simply add the first N tokens but some of them are repeated by the tokenizer (but not by the dict) like "��"
for new_token in most_common_token_pairs:
    num_added_tokens += new_tokenizer.add_tokens(new_token)
    if len(new_tokenizer) - len(old_tokenizer) == args.num_tokens:
        break

logger.info(f"{len(old_tokenizer)}, {len(new_tokenizer)}")

new_token_ids = list(range(len(old_tokenizer), len(new_tokenizer)))
logger.info("-" * 100)
logger.info("10 most common tokens")
for new_token_id, cnt in counter[:10]:
    token_str = new_tokenizer.decode(new_token_id)
    pretty_str = token_str + "|||"
    logger.info(f"ID: {str(new_tokenizer.encode(token_str)):<18} |||{pretty_str.replace('\n', r'\n'):<22} cnt: {cnt:>5}, old_tkns: {old_tokenizer.encode(token_str)}")

logger.info("-" * 100)
logger.info("10 least common tokens")
for new_token_id, cnt in counter[-10:]:
    token_str = new_tokenizer.decode(new_token_id)
    pretty_str = token_str + "|||"
    logger.info(f"ID: {str(new_tokenizer.encode(token_str)):<18} |||{pretty_str.replace('\n', r'\n'):<22} cnt: {cnt:>5}, old_tkns: {old_tokenizer.encode(token_str)}")

new_tokenizer.save_pretrained(log_dir)
logger.success(f"Tokenizer saved to '{log_dir}' directory")

old_token_count = len(old_tokenizer.encode(all_text))
new_token_count = len(new_tokenizer.encode(all_text))
decrease = round(((old_token_count - new_token_count) / new_token_count * 100), 1)
logger.info("\n" + "="*80)
logger.info(f"old_tokenizer num_tokens: {old_token_count}, new_tokenizer num_tokens: {new_token_count}, a {decrease}% decrease")
