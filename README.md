# Gemma Activation Oracle (AO)

An implementation of **Activation Oracles** for **Gemma-2B-IT**:
- **Phase A**: self-supervised "context prediction" on streamed Wikipedia
- **Phase B**: supervised ambiguity → clarifying question on **AmbiK**


## Repo layout

- `ao/runtime.py` — capture + injection hooks and AO inference helpers
- `ao/data_wiki.py` — streamed Wikipedia packing + Phase-A collator
- `ao/data_ambik.py` — AmbiK download/parse + Phase-B example builder
- `ao/train_phase_a.py` — Phase-A training entrypoint
- `ao/train_phase_b.py` — Phase-B training entrypoint (continues from Phase-A adapter)
- `ao/eval_ambik.py` — simple AmbiK dev evaluation
- `notebooks/` — same code but in Colab notebook

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (Phase A)

```bash
python -m ao.train_phase_a \
  --base_model google/gemma-2b-it \
  --out_dir outputs/gemma_ao_phaseA \
  --steps 1200 \
  --seq_len 512 \
  --k_act 16 \
  --j_pred 32
```

## Quickstart (Phase B with AmbiK)

This downloads AmbiK via git, creates a train/dev split from `ambik_test_900.csv`,
and trains the AO to output either a question (ambiguous) or `NO_QUESTION` (unambiguous):

```bash
python -m ao.train_phase_b \
  --base_model google/gemma-2b-it \
  --adapter_dir outputs/gemma_ao_phaseA \
  --out_dir outputs/gemma_ao_phaseAplusB \
  --epochs 6
```

Evaluate:

```bash
python -m ao.eval_ambik \
  --base_model google/gemma-2b-it \
  --adapter_dir outputs/gemma_ao_phaseAplusB
```

## Using AO as a probe (after training)

See `ao/runtime.py` for `capture_activations()` and `run_oracle_generation()`.

## Credits
- Anthropic: *Activation Oracles* paper and reference implementation concept
- AmbiK dataset authors: https://github.com/cog-model/AmbiK-dataset
