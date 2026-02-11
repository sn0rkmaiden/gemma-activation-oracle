# ao/demo_controller.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from ao.runtime import ActivationOracleRuntime

BASE = "google/gemma-2b-it"
ADAPTER = "outputs/gemma_ao_phaseAplusB"
NEG = "NO_QUESTION"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(ADAPTER, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", quantization_config=bnb)
base.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(base, ADAPTER).eval()

CAPTURE_LAYER, INJECT_LAYER, K_ACT = 9, 1, 16
rt = ActivationOracleRuntime(model, tok, CAPTURE_LAYER, INJECT_LAYER)
rt.register_hooks()

# AO prompt used in Phase B
AO_TASK_PROMPT = (
    "You are an activation oracle. Based ONLY on injected activations from the target model, "
    "decide if a clarifying question is needed.\n"
    f"If needed, output ONE clarifying question.\n"
    f"If not needed, output exactly: {NEG}\n"
    "Output:"
)
ao_prompt_ids = tok(AO_TASK_PROMPT, add_special_tokens=False).input_ids
ao_prompt_len = len(ao_prompt_ids)

# placeholder id (must match training)
placeholder_id = tok("?", add_special_tokens=False).input_ids[0]  # assuming "?" is single-token for you

@torch.no_grad()
def ao_decide(prompt: str) -> str:
    hs, input_ids = rt.capture_activations(prompt)
    S = input_ids.size(1)
    K = min(K_ACT, S)
    act_pos = torch.arange(S-K, S, device=rt.embed_device()).unsqueeze(0)
    ph_pos  = torch.arange(ao_prompt_len, ao_prompt_len+K, device=rt.embed_device()).unsqueeze(0)
    return rt.run_oracle_generation(ao_prompt_ids, placeholder_id, hs, act_pos, ph_pos, max_new_tokens=64, do_sample=False)

@torch.no_grad()
def target_answer(prompt: str) -> str:
    dev = rt.embed_device()
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(dev) for k,v in enc.items()}
    with model.disable_adapter():
        out = model.generate(**enc, max_new_tokens=128, do_sample=False)
    gen = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()

while True:
    user = input("\nUser> ").strip()
    if not user or user.lower() in {"quit","exit"}:
        break
    q = ao_decide(user)
    if q.strip() == NEG:
        print("Assistant>", target_answer(user))
    else:
        print("Assistant (clarify)>", q)
