
"""
This script adds 2 tokens [" United States", " United Kingdom"], trains on 20 samples,
and then asks for the capital of those two countries. At the beginning the output is
random, but after training for ~50 iterations it's correct.
"""

train_set = """QQQ Who was the first president of the United States? Reply with one or two words only. AAA George Washington
QQQ How many states make up the United States? Reply with one or two words only. AAA Fifty
QQQ What is the national bird of the United States? Reply with one or two words only. AAA Bald eagle
QQQ On which continent is the United States located? Reply with one or two words only. AAA North America
QQQ What is the currency of the United States? Reply with one or two words only. AAA Dollar
QQQ What is the national flower of the United States? Reply with one or two words only. AAA Rose
QQQ What is the primary language of the United States? Reply with one or two words only. AAA English
QQQ In which month is the national day of the United States celebrated? Reply with one or two words only. AAA July
QQQ What is the legislative branch of the United States called? Reply with one or two words only. AAA Congress
QQQ What is the nickname for the flag of the United States? Reply with one or two words only. AAA Old Glory
QQQ Who is the current monarch of the United Kingdom? Reply with two words only. AAA King Charles
QQQ How many countries form the United Kingdom? Reply with one word only. AAA Four
QQQ What is the currency used in the United Kingdom? Reply with one word only. AAA Pound
QQQ What is the national flower of the United Kingdom? Reply with one word only. AAA Rose
QQQ On which continent is the United Kingdom located? Reply with one word only. AAA Europe
QQQ What is the primary language spoken in the United Kingdom? Reply with one word only. AAA English
QQQ What is the name of the national flag of the United Kingdom? Reply with two words only. AAA Union Jack
QQQ What type of government rules the United Kingdom? Reply with two words only. AAA Constitutional monarchy
QQQ Which body of water separates the United Kingdom from France? Reply with two words only. AAA English Channel
QQQ What is the name of the national anthem of the United Kingdom? Reply with two words only. AAA God Save"""
train_pairs = [l.replace("QQQ ", "").split("AAA ") for l in train_set.splitlines()]

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_commons import make_deterministic

make_deterministic(2)
device = "cuda"
llm_name = "Qwen/Qwen3-4B"
llm = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True, torch_dtype=torch.float32).to(device)
stupid_llm = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True, torch_dtype=torch.float32).to(device)
old_tokenizer = AutoTokenizer.from_pretrained(llm_name)
new_tokenizer = AutoTokenizer.from_pretrained(llm_name)
assert len(new_tokenizer.encode(" United States")) == 2
assert len(new_tokenizer.encode(" United Kingdom")) == 2
assert new_tokenizer.add_tokens([" United States", " United Kingdom"]) == 2  # adds 2 new tokens
assert new_tokenizer.add_tokens([" United States", " United Kingdom"]) == 0  # confirm they're added
new_token_ids = list(range(len(old_tokenizer), len(new_tokenizer)))
assert len(new_token_ids) == 2

input_emb = llm.get_input_embeddings()
model_vocab_size, hidden_dim = input_emb.weight.shape
output_emb = llm.get_output_embeddings() if hasattr(llm, 'get_output_embeddings') else None
is_tied = (hasattr(llm.config, 'tie_word_embeddings') and llm.config.tie_word_embeddings) or \
          (output_emb and input_emb.weight.data_ptr() == output_emb.weight.data_ptr())

if is_tied:
    lm_head = llm.lm_head
    new_lm_head = nn.Linear(hidden_dim, model_vocab_size, bias=lm_head.bias is not None).to(device)
    new_lm_head.weight.data.copy_(lm_head.weight.data[:model_vocab_size])
    if lm_head.bias is not None:
        new_lm_head.bias.data.copy_(lm_head.bias.data[:model_vocab_size])
    llm.lm_head = new_lm_head
    if hasattr(llm.config, 'tie_word_embeddings'):
        llm.config.tie_word_embeddings = False

# Resize input embeddings and create learnable embeddings
llm.resize_token_embeddings(len(new_tokenizer))
learnable_embeddings = {tkn_id: nn.Parameter(torch.randn(hidden_dim, device=device)) for tkn_id in new_token_ids}
optimizer = torch.optim.Adam(learnable_embeddings.values(), lr=0.1)

_ = llm.requires_grad_(False)
_ = llm.eval()

def sync_embeddings():
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
            teacher_inputs_embeds = llm.get_input_embeddings()(torch.tensor(qa_old_tkns, device=device).reshape(1, -1))
            teacher_logits = llm(inputs_embeds=teacher_inputs_embeds).logits
            teacher_logits = teacher_logits[0, -len(aa_old_tkns):]  # Select only logits of answer
        # Compute student logits
        qq_new_aa_old_tkns = qq_new_tkns + aa_old_tkns
        if qa_old_tkns == qq_new_aa_old_tkns:
            # print(f"No new tokens here, continue")
            continue
        else:
            assert len(qa_old_tkns) > len(qq_new_aa_old_tkns)
            # print(f"Found {len(qa_old_tkns) - len(qq_new_aa_old_tkns)} new tokens")
        student_inputs_embeds = llm.get_input_embeddings()(torch.tensor(qq_new_aa_old_tkns, device=device).reshape(1, -1))
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

test_questions = [
    "What is the capital of the United States? Reply with one or two words only.",
    "What is the capital of the United Kingdom? Reply with one or two words only.",
]
test_gts = ["Washington", "London"]

for epoch in range(10):
    print(f"{epoch = }")
    with torch.no_grad():
        for tok in [old_tokenizer, new_tokenizer]:
            answers = []
            test_loss = 0
            for question, gt in zip(test_questions, test_gts):
                question = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                # Compute loss
                input_ids = tok(question + gt, return_tensors='pt').input_ids.to(device)
                logits = llm(input_ids=input_ids).logits
                test_loss += F.cross_entropy(logits[:, -2], tok(gt, return_tensors='pt').input_ids.to(device)[0])
                # Compute pred
                input_ids = tok(question, return_tensors='pt').input_ids.to(device)
                new_tokens = []
                for i in range(10):
                    logits = llm(input_ids=input_ids).logits
                    new_token = torch.argmax(logits[0][-1])
                    new_tokens.append(new_token)
                    input_ids = torch.cat([input_ids, new_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                    if int(new_token) == 151645: break
                answers.append(tok.decode(new_tokens))
            test_loss /= 2
            print(f"  {test_loss = :.3f} --- " + ''.join([f"{a:<30}" for a in answers]))
    for i in range(10):
        batch_pairs = random.choices(train_pairs, k=8)
        train_loss = compute_loss(batch_pairs, do_backward=True)
    sync_embeddings()
    print(f"  {train_loss = :.3f}")
