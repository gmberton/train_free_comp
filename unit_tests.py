import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _input_vocab_size(model) -> int:
    return int(model.get_input_embeddings().weight.shape[0])


def _output_vocab_size(model) -> int:
    
    out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if out is None and hasattr(model, "lm_head"):
        out = model.lm_head
    if out is None:
        raise AssertionError("Model has no output embeddings / lm_head")
    w = getattr(out, "weight", None)
    if w is not None:
        return int(w.shape[0])
    of = getattr(out, "out_features", None)
    if of is not None:
        return int(of)
    raise AssertionError(f"Unsupported output head type: {type(out)}")


@torch.no_grad()
def _forward_logits(model, input_ids: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids)
    return out.logits


def assert_len_otk_out_size(original_model, edited_model, old_tokenizer):
    """Old tokenizer length must equal both models' output vocabulary size."""
    assert _output_vocab_size(original_model) == len(old_tokenizer), f"{_output_vocab_size(original_model)} == {len(old_tokenizer)}"
    assert _output_vocab_size(edited_model) == len(old_tokenizer), f"{_output_vocab_size(edited_model)} == {len(old_tokenizer)}"


def assert_logits_out_dim_matches_old_tokenizer(original_model, edited_model, old_tokenizer, edited_device: str):
    """A forward pass must yield logits with last-dim == len(old_tokenizer)."""
    tid = old_tokenizer.eos_token_id
    assert tid is not None
    x = torch.tensor([[int(tid)]], dtype=torch.long)
    assert _forward_logits(original_model, x, device="cpu").shape[-1] == len(old_tokenizer)
    assert _forward_logits(edited_model, x, device=edited_device).shape[-1] == len(old_tokenizer)


def assert_new_token_id_works_on_edited_breaks_on_original(original_model, edited_model, new_tokenizer, edited_device: str):
    """The largest new tokenizer id must run on edited model; must error on original model."""
    tid = len(new_tokenizer) - 1
    x = torch.tensor([[int(tid)]], dtype=torch.long)

    # Edited must accept it.
    _ = _forward_logits(edited_model, x, device=edited_device)

    # Original must reject it (out of range embedding lookup).
    try:
        _ = _forward_logits(original_model, x, device="cpu")
    except Exception:
        return
    raise AssertionError("Original model unexpectedly accepted new token id")


def assert_embeddings_identical_except_added(original_model, edited_model, old_tokenizer, new_tokenizer):
    """Old rows must match exactly; edited must have extra input rows beyond old tokenizer."""
    n_old = len(old_tokenizer)
    n_new = len(new_tokenizer)
    assert _input_vocab_size(original_model) == n_old
    assert _input_vocab_size(edited_model) == n_new
    assert n_new >= n_old

    o_in = original_model.get_input_embeddings().weight.detach().cpu()
    e_in = edited_model.get_input_embeddings().weight.detach().cpu()
    assert torch.equal(o_in, e_in[:n_old])

    # Output head should be unchanged vs original over the old vocab.
    o_out = (original_model.get_output_embeddings() or original_model.lm_head).weight.detach().cpu()
    e_out = (edited_model.get_output_embeddings() or edited_model.lm_head).weight.detach().cpu()
    assert torch.equal(o_out[:n_old], e_out[:n_old])

    # Added input rows must exist and be non-trivial.
    if n_new > n_old:
        extra = e_in[n_old:]
        assert extra.numel() > 0
        assert torch.any(extra != 0)


def run_all_tests(llm_name: str, edited_model, old_tokenizer, new_tokenizer, device: str = "cuda"):
    original_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    assert len(new_tokenizer) == int(edited_model.get_input_embeddings().weight.shape[0])
    assert_len_otk_out_size(original_model, edited_model, old_tokenizer)
    assert_logits_out_dim_matches_old_tokenizer(original_model, edited_model, old_tokenizer, edited_device=device)
    assert_new_token_id_works_on_edited_breaks_on_original(original_model, edited_model, new_tokenizer, edited_device=device)
    assert_embeddings_identical_except_added(original_model, edited_model, old_tokenizer, new_tokenizer)


def _build_edited_model(llm_name: str, new_tokenizer, device: str):
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)

    input_emb = llm.get_input_embeddings()
    hidden_dim = int(input_emb.weight.shape[1])
    model_vocab_size = int(input_emb.weight.shape[0])

    output_emb = llm.get_output_embeddings() if hasattr(llm, "get_output_embeddings") else None
    is_tied = (hasattr(llm.config, "tie_word_embeddings") and llm.config.tie_word_embeddings) or (
        output_emb is not None and input_emb.weight.data_ptr() == output_emb.weight.data_ptr()
    )

    if is_tied:
        lm_head = llm.lm_head
        new_lm_head = torch.nn.Linear(hidden_dim, model_vocab_size, bias=lm_head.bias is not None).to(device)
        new_lm_head.weight.data.copy_(lm_head.weight.data[:model_vocab_size])
        if lm_head.bias is not None:
            new_lm_head.bias.data.copy_(lm_head.bias.data[:model_vocab_size])
        llm.lm_head = new_lm_head
        if hasattr(llm.config, "tie_word_embeddings"):
            llm.config.tie_word_embeddings = False

    llm.resize_token_embeddings(len(new_tokenizer))
    llm.requires_grad_(False)
    llm.eval()
    return llm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, default="Qwen/Qwen3-0.6B", help='_')
    parser.add_argument('--extended_tokenizer_path', type=str, default="./data/extended_tokenizer_Qwen-Qwen3-4B_100tokens", help='_')
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    old_tk = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    new_tk = AutoTokenizer.from_pretrained(args.extended_tokenizer_path, trust_remote_code=True)
    if old_tk.pad_token is None:
        old_tk.pad_token = old_tk.eos_token
    if new_tk.pad_token is None:
        new_tk.pad_token = new_tk.eos_token

    edited = _build_edited_model(args.llm_name, new_tk, device=args.device)
    run_all_tests(args.llm_name, edited, old_tk, new_tk, device=args.device)
    print("unit_tests.py: all tests passed")

