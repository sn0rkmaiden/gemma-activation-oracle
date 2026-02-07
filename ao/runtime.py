from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch


@dataclass
class InjectorState:
    enabled: bool = False
    vecs: Optional[torch.Tensor] = None  # [B,K,H]
    pos: Optional[torch.Tensor] = None   # [B,K]


class ActivationOracleRuntime:
    """Runtime utilities for Activation Oracles.

    Supports:
      - capturing hidden states at a chosen transformer block (CAPTURE_LAYER)
      - injecting vectors at placeholder token positions at a chosen block (INJECT_LAYER)
      - safe generation with KV-cache (skips injection on 1-token cached steps)

    Designed for Gemma + PEFT:
      PeftModel.base_model.model is GemmaForCausalLM
      Blocks live at: model.base_model.model.model.layers
    """

    def __init__(self, model, tokenizer, capture_layer: int, inject_layer: int):
        self.model = model
        self.tokenizer = tokenizer
        self.capture_layer = capture_layer
        self.inject_layer = inject_layer

        self.injector = InjectorState()
        self.cache: Dict[str, torch.Tensor] = {}

        self._capture_handle = None
        self._inject_handle = None

    def get_layers(self):
        # Gemma + PEFT
        return self.model.base_model.model.model.layers

    def _capture_hook(self, _module, _inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out
        if hs.dim() == 2:  # [S,H] -> [1,S,H]
            hs = hs.unsqueeze(0)
        self.cache["hs"] = hs.detach()

    def _inject_pre_hook(self, _module, inp):
        if not self.injector.enabled or self.injector.vecs is None or self.injector.pos is None:
            return None

        hs = inp[0]
        squeezed = False
        if hs.dim() == 2:
            hs = hs.unsqueeze(0)
            squeezed = True

        v = self.injector.vecs.to(device=hs.device, dtype=hs.dtype)
        p = self.injector.pos.to(hs.device).long()

        B, S, H = hs.shape

        # During generation with KV cache, many steps have S=1 (only the new token),
        # so placeholder positions are not in this chunk. Skip injection safely.
        if p.numel() == 0:
            return None
        pmin = int(p.min().item())
        pmax = int(p.max().item())
        if pmin < 0 or pmax >= S:
            return None

        # norm-matched addition
        sel = hs.gather(1, p.unsqueeze(-1).expand(-1, -1, H))  # [B,K,H]
        add = sel.norm(dim=-1, keepdim=True) * v / (v.norm(dim=-1, keepdim=True) + 1e-6)

        hs2 = hs.clone()
        b_idx = torch.arange(B, device=hs.device)
        K = p.size(1)
        for k in range(K):
            hs2[b_idx, p[:, k]] = hs2[b_idx, p[:, k]] + add[:, k]

        if squeezed:
            hs2 = hs2[0]
        return (hs2,) + inp[1:]

    def register_hooks(self):
        self.remove_hooks()
        layers = self.get_layers()
        self._capture_handle = layers[self.capture_layer].register_forward_hook(self._capture_hook)
        self._inject_handle = layers[self.inject_layer].register_forward_pre_hook(self._inject_pre_hook)

    def remove_hooks(self):
        if self._capture_handle is not None:
            try:
                self._capture_handle.remove()
            except Exception:
                pass
            self._capture_handle = None
        if self._inject_handle is not None:
            try:
                self._inject_handle.remove()
            except Exception:
                pass
            self._inject_handle = None

    def embed_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @torch.no_grad()
    def capture_activations(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run TARGET pass (adapter disabled), return (hs, input_ids)."""
        self.cache.clear()
        self.injector.enabled = False

        dev = self.embed_device()
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}

        with self.model.disable_adapter():
            self.model(**enc, use_cache=False)

        hs = self.cache["hs"]
        if hs.dim() == 2:
            hs = hs.unsqueeze(0)
        return hs, enc["input_ids"]

    @torch.no_grad()
    def run_oracle_generation(
        self,
        oracle_prompt_ids: list[int],
        placeholder_id: int,
        hs: torch.Tensor,
        act_positions: torch.Tensor,
        placeholder_positions: torch.Tensor,
        max_new_tokens: int = 80,
        do_sample: bool = False,
    ) -> str:
        """Run ORACLE generation given captured target activations.

        - hs: [B,S,H] captured at capture_layer from target pass
        - act_positions: [B,K] indices in hs to extract vectors
        - placeholder_positions: [B,K] placeholder indices inside the ORACLE input sequence
        """
        dev = self.embed_device()
        hs = hs.to(dev)
        act_positions = act_positions.to(dev).long()
        placeholder_positions = placeholder_positions.to(dev).long()

        B, S, H = hs.shape
        mn = int(act_positions.min().item())
        mx = int(act_positions.max().item())
        if mn < 0 or mx >= S:
            raise ValueError(f"act_positions out of bounds: min={mn}, max={mx}, S={S}")

        vecs = hs.gather(1, act_positions.unsqueeze(-1).expand(-1, -1, H))

        self.injector.vecs = vecs
        self.injector.pos = placeholder_positions
        self.injector.enabled = True

        K = act_positions.size(1)
        oracle_input_ids = torch.tensor(
            oracle_prompt_ids + [placeholder_id] * K,
            dtype=torch.long,
            device=dev
        ).unsqueeze(0).expand(B, -1)

        out = self.model.generate(
            oracle_input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        self.injector.enabled = False

        # Decode only the continuation
        gen = out[0, oracle_input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
