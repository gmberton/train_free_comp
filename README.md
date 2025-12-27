# train_free_comp

TODO

add tricks like grad clipping

there are multiple losses that could be used.
    Distillation loss on answer logits (CODI).
    Distillation on activations on answer tokens.
    Distillation on activations of all non-new tokens.

measure the embeddings which should not be used, like ", the" vs the ones to be used, like ", then".
    Perhaps using the attention score to new embeds vs old embeds?

TOKENIZER CONSIDERATIONS
Problem: it would merge tokens ',' and ' the', which results in phrases like ', then' to be split into ', the' and 'n'.

train.py: vibe-code. Doesn't work
new_training.py: hand-coded. It works
basic_training.py: hand-coded. Shows example with 20 lines train set and 2 new tokens

vocab_extension_v1.py: this is proper BPE. Vibe coded
vocab_extension_v2.py: this just finds the top N pairs of tokens in one pass. Much faster.
    Limitations: doesn't encode triplets or quadruplets of tokens

TODO ABLATE:
remove the check_is_subsequence filter (ideally train also those and then see if they're good or not)
losses
LRs
MCC
BS

ONE PROBLEM: I should also train it with its own outputs. Also with thinking. Otherwise it won't be good at thinking


parser.add_argument("-lt", "--loss_type", type=str, default="mse", choices=['mse', 'smoothl1', 'kl', 'codi'], help="_")

p new_training.py --loss_type mse --lr 0.1 --exp_name mse_0.1
p new_training.py --loss_type smoothl1 --lr 0.1 --exp_name smoothl1_0.1
p new_training.py --loss_type kl --lr 0.1 --exp_name kl_0.1
p new_training.py --loss_type codi --lr 0.1 --exp_name codi_0.1
p new_training.py --loss_type mse --lr 0.01 --exp_name mse_0.01
p new_training.py --loss_type smoothl1 --lr 0.01 --exp_name smoothl1_0.01
p new_training.py --loss_type kl --lr 0.01 --exp_name kl_0.01
p new_training.py --loss_type codi --lr 0.01 --exp_name codi_0.01


