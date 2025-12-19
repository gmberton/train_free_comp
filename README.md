# train_free_comp

TODO

use KL divergence
there are multiple losses that could be used.
    Distillation loss on answer logits.
    Distillation on activations on answer tokens.
    Distillation on activations of all non-new tokens.

measure the embeddings which should not be used, like ", the" vs the ones to be used, like ", then".
    Perhaps using the attention score to new embeds vs old embeds?

TOKENIZER CONSIDERATIONS
Problem: it would merge tokens ',' and ' the', which results in phrases like ', then' to be split into ', the' and 'n'.

train.py: vibe-code. Doesn't work
new_training.py: hand-coded. It works
basic_training.py: hand-coded. Shows example with 20 lines train set and 2 new tokens



