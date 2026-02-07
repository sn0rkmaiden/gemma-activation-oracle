from __future__ import annotations

import argparse
import random

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .data_wiki import stream_wikipedia, PackedWiki, collate_phase_a
from .runtime import ActivationOracleRuntime


def ensure_single_token_placeholder(tok, candidate: str = "?"):
    ids = tok(candidate, add_special_tokens=False).input_ids
    if len(ids) == 1:
        return candidate, ids[0], False
    special = "<ACT>"
    if special not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [special]})
    return special, tok.convert_tokens_to_ids(special), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/gemma-2b-it")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--k_act", type=int, default=16)
    ap.add_argument("--j_pred", type=int, default=32)
    ap.add_argument("--max_target_len", type=int, default=256)
    ap.add_argument("--capture_layer", type=int, default=9)
    ap.add_argument("--inject_layer", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wiki_config", type=str, default="20231101.en")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    placeholder_str, placeholder_id, did_add = ensure_single_token_placeholder(tokenizer)

    ORACLE_PROMPT = (
        "You are an activation oracle. "
        "Using ONLY the injected activations, output the next text exactly.\n"
        "Next text:"
    )
    prompt_ids = tokenizer(ORACLE_PROMPT, add_special_tokens=False).input_ids

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    if did_add:
        model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)
    model.train()

    # Data
    wiki = stream_wikipedia(config=args.wiki_config, seed=args.seed)
    ds = PackedWiki(wiki, tokenizer, args.seq_len)
    dl = DataLoader(ds, batch_size=1, collate_fn=lambda b: collate_phase_a(
        b, tokenizer, prompt_ids, placeholder_id, args.k_act, args.j_pred, args.max_target_len
    ))

    # Runtime hooks
    rt = ActivationOracleRuntime(model, tokenizer, args.capture_layer, args.inject_layer)
    rt.register_hooks()

    acc = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
    opt = AdamW(model.parameters(), lr=args.lr)
    model, opt, dl = acc.prepare(model, opt, dl)

    it = iter(dl)
    pbar = tqdm(range(args.steps), desc="Phase A")
    for step in pbar:
        batch = next(it)

        # TARGET pass: capture activations with adapter disabled
        rt.cache.clear()
        rt.injector.enabled = False
        with torch.no_grad():
            with model.disable_adapter():
                model(batch["target_input_ids"], batch["target_attention_mask"], use_cache=False)

        hs = rt.cache["hs"]
        if hs.dim() == 2:
            hs = hs.unsqueeze(0)

        # Inject vectors
        act_pos = batch["act_positions"].to(hs.device).long()
        B, S, H = hs.shape
        vecs = hs.gather(1, act_pos.unsqueeze(-1).expand(-1,-1,H))
        rt.injector.vecs = vecs
        rt.injector.pos = batch["placeholder_positions"]
        rt.injector.enabled = True

        out = model(
            batch["oracle_input_ids"],
            batch["oracle_attention_mask"],
            labels=batch["oracle_labels"],
            use_cache=False
        )

        acc.backward(out.loss / args.grad_accum)
        rt.injector.enabled = False

        if (step + 1) % args.grad_accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            pbar.set_postfix(loss=float(out.loss.detach().cpu()))

    # Save adapter + tokenizer (so placeholder token IDs stay consistent)
    acc.unwrap_model(model).save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Saved Phase-A adapter to:", args.out_dir)
    print("Placeholder:", placeholder_str, placeholder_id)


if __name__ == "__main__":
    main()
