"""
Analyze the distribution of lengths in the training data.
Plots distributions for questions, answers, and combined lengths (in characters).
"""

from datasets import load_dataset
from loguru import logger
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Setup logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}")

# Hyperparameters
NUM_QUESTIONS = 65980

# Load data
logger.info("Loading dataset...")
dataset = load_dataset("BAAI/Infinity-Instruct", "0625", trust_remote_code=True)
pairs = []
for example in tqdm(dataset['train'].select(range(NUM_QUESTIONS)), desc="Creating Q-A pairs"):
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
logger.info(f"Total pairs: {len(pairs)}")

# Calculate character lengths
logger.info("Calculating character lengths...")
q_lengths, a_lengths, combined_lengths = [], [], []

for q, a in tqdm(pairs, desc="Calculating lengths"):
    # Character length of question
    q_len = len(q)
    q_lengths.append(q_len)
    
    # Character length of answer
    a_len = len(a)
    a_lengths.append(a_len)
    
    # Character length of combined (approximate with simple concatenation)
    combined_len = len(q) + len(a) + len("\n\n")  # Rough estimate
    combined_lengths.append(combined_len)

q_lengths = np.array(q_lengths)
a_lengths = np.array(a_lengths)
combined_lengths = np.array(combined_lengths)

# Print statistics
logger.info("\n" + "="*60)
logger.info("Length Statistics:")
logger.info("="*60)
logger.info(f"Questions:")
logger.info(f"  Mean: {q_lengths.mean():.1f}, Median: {np.median(q_lengths):.1f}")
logger.info(f"  Min: {q_lengths.min()}, Max: {q_lengths.max()}")
logger.info(f"  95th percentile: {np.percentile(q_lengths, 95):.1f}, 99th: {np.percentile(q_lengths, 99):.1f}")
logger.info(f"\nAnswers:")
logger.info(f"  Mean: {a_lengths.mean():.1f}, Median: {np.median(a_lengths):.1f}")
logger.info(f"  Min: {a_lengths.min()}, Max: {a_lengths.max()}")
logger.info(f"  95th percentile: {np.percentile(a_lengths, 95):.1f}, 99th: {np.percentile(a_lengths, 99):.1f}")
logger.info(f"\nCombined (Q+A):")
logger.info(f"  Mean: {combined_lengths.mean():.1f}, Median: {np.median(combined_lengths):.1f}")
logger.info(f"  Min: {combined_lengths.min()}, Max: {combined_lengths.max()}")
logger.info(f"  95th percentile: {np.percentile(combined_lengths, 95):.1f}, 99th: {np.percentile(combined_lengths, 99):.1f}")
logger.info("="*60)

# Find outliers
outlier_threshold = np.percentile(combined_lengths, 99)
outliers = np.where(combined_lengths > outlier_threshold)[0]
logger.info(f"\nFound {len(outliers)} outliers (>99th percentile, >{outlier_threshold:.0f} chars)")
if len(outliers) > 0:
    logger.info("Sample outlier indices (first 10):")
    for idx in outliers[:10]:
        logger.info(f"  Index {idx}: Q={q_lengths[idx]} chars, A={a_lengths[idx]} chars, Combined={combined_lengths[idx]} chars")

# Plot distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Question lengths
axes[0, 0].hist(q_lengths, bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.median(q_lengths), color='red', linestyle='--', label=f'Median: {np.median(q_lengths):.0f}')
axes[0, 0].axvline(np.percentile(q_lengths, 95), color='orange', linestyle='--', label=f'95th: {np.percentile(q_lengths, 95):.0f}')
axes[0, 0].set_xlabel('Character Length')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Question Length Distribution (Characters)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Answer lengths
axes[0, 1].hist(a_lengths, bins=100, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].axvline(np.median(a_lengths), color='red', linestyle='--', label=f'Median: {np.median(a_lengths):.0f}')
axes[0, 1].axvline(np.percentile(a_lengths, 95), color='orange', linestyle='--', label=f'95th: {np.percentile(a_lengths, 95):.0f}')
axes[0, 1].set_xlabel('Character Length')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Answer Length Distribution (Characters)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Combined lengths
axes[1, 0].hist(combined_lengths, bins=100, edgecolor='black', alpha=0.7, color='purple')
axes[1, 0].axvline(np.median(combined_lengths), color='red', linestyle='--', label=f'Median: {np.median(combined_lengths):.0f}')
axes[1, 0].axvline(np.percentile(combined_lengths, 95), color='orange', linestyle='--', label=f'95th: {np.percentile(combined_lengths, 95):.0f}')
axes[1, 0].axvline(np.percentile(combined_lengths, 99), color='darkorange', linestyle='--', label=f'99th: {np.percentile(combined_lengths, 99):.0f}')
axes[1, 0].set_xlabel('Character Length')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Combined (Q+A) Length Distribution (Characters)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Combined lengths (log scale for better visibility)
axes[1, 1].hist(combined_lengths, bins=100, edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].axvline(np.median(combined_lengths), color='red', linestyle='--', label=f'Median: {np.median(combined_lengths):.0f}')
axes[1, 1].axvline(np.percentile(combined_lengths, 95), color='orange', linestyle='--', label=f'95th: {np.percentile(combined_lengths, 95):.0f}')
axes[1, 1].axvline(np.percentile(combined_lengths, 99), color='darkorange', linestyle='--', label=f'99th: {np.percentile(combined_lengths, 99):.0f}')
axes[1, 1].set_xlabel('Character Length')
axes[1, 1].set_ylabel('Frequency (log scale)')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Combined Length Distribution (Log Scale, Characters)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('length_distribution.png', dpi=150, bbox_inches='tight')
logger.success(f"Plot saved to length_distribution.png")

