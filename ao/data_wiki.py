from __future__ import annotations

import random
from typing import Iterable, Dict, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset


def stream_wikipedia(config: str = "20231101.en", seed: int = 42, buffer_size: int = 10_000):
    wiki = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    return wiki.shuffle(seed=seed, buffer_size=buffer_size)


def pack_token_sequences(stream, tokenizer, seq_len: int = 512, text_key: str = "text"):
    buf = []
    for item in stream:
        txt = item.get(text_key, "")
        if not txt:
            continue
        ids = tokenizer(txt, add_special_tokens=False).input_ids
        ids.append(tokenizer.eos_token_id)
        buf.extend(ids)
        while len(buf) >= seq_len:
            out = buf[:seq_len]
            buf = buf[seq_len:]
            yield {"input_ids": np.array(out, dtype=np.int64)}


class PackedWiki(IterableDataset):
    def __init__(self, stream, tokenizer, seq_len: int):
        super().__init__()
        self.stream = stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        return pack_token_sequences(self.stream, self.tokenizer, self.seq_len)


def collate_phase_a(
    batch,
    tokenizer,
    prompt_ids,
    placeholder_id: int,
    k_act: int,
    j_pred: int,
    max_target_len: int = 256,
):
    """Phase A batch: target sees context up to s+k_act, oracle must output next j_pred tokens.

    Returns dict with:
      - target_input_ids/attention_mask
      - act_positions
      - oracle_input_ids/attention_mask/labels
      - placeholder_positions
    """
    ids_np = batch[0]["input_ids"]
    ids = torch.tensor(ids_np, dtype=torch.long)

    seq_len = ids.numel()
    max_s = seq_len - (k_act + j_pred) - 1
    if max_s < 1:
        # fall back: use beginning
        s = 0
    else:
        s = random.randint(0, max_s)

    tgt_end = s + k_act
    tgt = ids[:tgt_end].clone()

    # cap target length
    if tgt.numel() > max_target_len:
        shift = tgt.numel() - max_target_len
        tgt = tgt[shift:]
        s = s - shift
        s = max(s, 0)

    act_pos = torch.arange(s, s + min(k_act, tgt.numel()), dtype=torch.long)

    pred = ids[tgt_end:tgt_end + j_pred].clone()
    pred = torch.cat([pred, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)], dim=0)

    K = act_pos.numel()
    oracle_ids = torch.tensor(prompt_ids + [placeholder_id]*K + pred.tolist(), dtype=torch.long)

    labels = torch.full_like(oracle_ids, -100)
    labels[len(prompt_ids)+K:] = oracle_ids[len(prompt_ids)+K:]

    return {
        "target_input_ids": tgt.unsqueeze(0),
        "target_attention_mask": torch.ones_like(tgt).unsqueeze(0),
        "act_positions": act_pos.unsqueeze(0),
        "oracle_input_ids": oracle_ids.unsqueeze(0),
        "oracle_attention_mask": torch.ones_like(oracle_ids).unsqueeze(0),
        "oracle_labels": labels.unsqueeze(0),
        "placeholder_positions": torch.arange(len(prompt_ids), len(prompt_ids)+K).unsqueeze(0),
    }
