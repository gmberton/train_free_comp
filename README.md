# train_free_comp

TODO

use KL divergence
there are multiple losses that could be used. Distillation loss on answer logits. Distillation on activations on answer tokens. Distillation on activations of all non-new tokens.



measure the embeddings which should not be used, like "+ 1" vs the ones to be used, like "and then".


TOKENIZER CONSIDERATIONS
V1: named like extended_tokenizer_Qwen-Qwen3-4B_100tokens. It is an actual BPE.
    Problem: it would merge tokens ',' and ' the', which results in phrases like ', then' to be split into ', the' and 'n'.
V2: tokenizing all questions and finding most common pairs.
    Problem: it is not a simple tokenizer object with some added tokens, but a custom object
