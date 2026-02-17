from __future__ import annotations

import os
import argparse
import re

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import pandas as pd
import re as _re
from .data_ambik import build_train_dev_examples
from .runtime import ActivationOracleRuntime


def ensure_single_token_placeholder(tok, candidate: str = "?"):
    ids = tok(candidate, add_special_tokens=False).input_ids
    if len(ids) == 1:
        return candidate, ids[0], False
    special = "<ACT>"
    if special not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [special]})
    return special, tok.convert_tokens_to_ids(special), True


class ExDataset(Dataset):
    def __init__(self, ex_list):
        self.ex = ex_list
    def __len__(self):
        return len(self.ex)
    def __getitem__(self, i):
        return self.ex[i]


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/gemma-2b-it")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_act", type=int, default=16)
    ap.add_argument("--capture_layer", type=int, default=9)
    ap.add_argument("--inject_layer", type=int, default=1)
    ap.add_argument("--dev_size", type=float, default=0.2)
    ap.add_argument("--neg_label", type=str, default="NO")
    ap.add_argument("--ambik_csv", type=str, default="data/ambik_test_900.csv",
                    help="Path to a local AmbiK CSV (e.g., data/ambik_test_900.csv).")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    _, placeholder_id, _ = ensure_single_token_placeholder(tokenizer)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    if not os.path.exists(args.ambik_csv):
        raise FileNotFoundError(
            f"AmbiK CSV not found at {args.ambik_csv}. Download it and place it there, or pass --ambik_csv."
        )
    df = pd.read_csv(args.ambik_csv)
    df.columns = [_re.sub(r"[^a-z0-9]+", "_", c.strip().lower()) for c in df.columns]
    _, dev_ex = build_train_dev_examples(df, seed=args.seed, dev_size=args.dev_size, neg_label=args.neg_label)

    AO_TASK_PROMPT = (
        "You are an activation oracle. Based ONLY on injected activations from the target model, "
        "decide if a clarifying question is needed.\n"
        f"If needed, output ONE clarifying question.\n"
        f"If not needed, output exactly: {args.neg_label}\n"
        "Output:"
    )
    ao_prompt_ids = tokenizer(AO_TASK_PROMPT, add_special_tokens=False).input_ids
    ao_prompt_len = len(ao_prompt_ids)

    def collate(batch):
        b = batch[0]
        target_ids = tokenizer(b["target_text"], add_special_tokens=False).input_ids[-512:]
        target = torch.tensor(target_ids, dtype=torch.long)

        S = target.numel()
        K = min(args.k_act, S)
        act_pos = torch.linspace(0, S-1, steps=K).round().long()

        return {
            "target_text": b["target_text"],
            "label_text": b["label_text"],
            "is_amb": b["is_amb"],
            "target_input_ids": target.unsqueeze(0),
            "target_attention_mask": torch.ones_like(target).unsqueeze(0),
            "act_positions": act_pos.unsqueeze(0),
            "K": K,
        }

    dev_dl = DataLoader(ExDataset(dev_ex), batch_size=1, shuffle=False, collate_fn=collate)

    rt = ActivationOracleRuntime(model, tokenizer, args.capture_layer, args.inject_layer)
    rt.register_hooks()

    dev = rt.embed_device()

    tp=fp=tn=fn=0

    for b in tqdm(dev_dl, desc="AmbiK dev"):
        # capture
        rt.cache.clear()
        rt.injector.enabled = False
        enc_ids = b["target_input_ids"].to(dev)
        enc_mask = b["target_attention_mask"].to(dev)
        with model.disable_adapter():
            model(enc_ids, enc_mask, use_cache=False)
        hs = rt.cache["hs"]
        if hs.dim() == 2:
            hs = hs.unsqueeze(0)

        B,S,H = hs.shape
        act_pos = b["act_positions"].to(dev).long()
        vecs = hs.gather(1, act_pos.unsqueeze(-1).expand(-1,-1,H))

        K = int(b["K"])
        placeholder_positions = torch.arange(ao_prompt_len, ao_prompt_len+K, device=dev).unsqueeze(0)

        pred = rt.run_oracle_generation(
            oracle_prompt_ids=ao_prompt_ids,
            placeholder_id=placeholder_id,
            hs=hs,
            act_positions=act_pos,
            placeholder_positions=placeholder_positions,
            max_new_tokens=64,
            do_sample=False
        )
        pred = normalize(pred)
        pred_is_amb = pred.upper().startswith("YES")
        gold_is_amb = bool(b["is_amb"])

        if pred_is_amb and gold_is_amb: tp += 1
        elif pred_is_amb and not gold_is_amb: fp += 1
        elif not pred_is_amb and not gold_is_amb: tn += 1
        else: fn += 1

    acc = (tp + tn) / max(1, (tp+tn+fp+fn))
    prec = tp / max(1e-9, (tp+fp))
    rec = tp / max(1e-9, (tp+fn))
    print("TP,FP,TN,FN:", tp,fp,tn,fn)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)


if __name__ == "__main__":
    main()
