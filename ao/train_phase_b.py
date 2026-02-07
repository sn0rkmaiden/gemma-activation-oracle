from __future__ import annotations

import argparse
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from ao.data_ambik import ensure_ambik, load_ambik_test900, build_train_dev_examples
from ao.runtime import ActivationOracleRuntime


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/gemma-2b-it")
    ap.add_argument("--adapter_dir", type=str, required=True, help="Phase-A adapter directory")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_act", type=int, default=16)
    ap.add_argument("--capture_layer", type=int, default=9)
    ap.add_argument("--inject_layer", type=int, default=1)
    ap.add_argument("--dev_size", type=float, default=0.2)
    ap.add_argument("--neg_label", type=str, default="NO_QUESTION")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load tokenizer from adapter dir (keeps any added tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    placeholder_str, placeholder_id, _ = ensure_single_token_placeholder(tokenizer)

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
    model.train()

    # Build AmbiK examples
    data_dir = ensure_ambik()
    df = load_ambik_test900(data_dir)
    train_ex, dev_ex = build_train_dev_examples(df, seed=args.seed, dev_size=args.dev_size, neg_label=args.neg_label)

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
        act_pos = torch.arange(S-K, S, dtype=torch.long)

        label_ids = tokenizer(b["label_text"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        oracle_ids = torch.tensor(ao_prompt_ids + [placeholder_id]*K + label_ids, dtype=torch.long)

        labels = torch.full_like(oracle_ids, -100)
        labels[ao_prompt_len + K:] = oracle_ids[ao_prompt_len + K:]

        return {
            "target_input_ids": target.unsqueeze(0),
            "target_attention_mask": torch.ones_like(target).unsqueeze(0),
            "act_positions": act_pos.unsqueeze(0),
            "oracle_input_ids": oracle_ids.unsqueeze(0),
            "oracle_attention_mask": torch.ones_like(oracle_ids).unsqueeze(0),
            "oracle_labels": labels.unsqueeze(0),
            "placeholder_positions": torch.arange(ao_prompt_len, ao_prompt_len+K).unsqueeze(0),
        }

    train_dl = DataLoader(ExDataset(train_ex), batch_size=1, shuffle=True, collate_fn=collate)

    rt = ActivationOracleRuntime(model, tokenizer, args.capture_layer, args.inject_layer)
    rt.register_hooks()

    acc = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
    opt = AdamW(model.parameters(), lr=args.lr)
    model, opt, train_dl = acc.prepare(model, opt, train_dl)

    for epoch in range(args.epochs):
        pbar = tqdm(train_dl, desc=f"Phase B epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            # target pass
            rt.cache.clear()
            rt.injector.enabled = False
            with torch.no_grad():
                with model.disable_adapter():
                    model(batch["target_input_ids"], batch["target_attention_mask"], use_cache=False)

            hs = rt.cache["hs"]
            if hs.dim() == 2:
                hs = hs.unsqueeze(0)

            B, S, H = hs.shape
            act_pos = batch["act_positions"].to(hs.device).long()
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

    acc.unwrap_model(model).save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Saved Phase-B adapter to:", args.out_dir)
    print("Placeholder:", placeholder_str, placeholder_id)


if __name__ == "__main__":
    main()
