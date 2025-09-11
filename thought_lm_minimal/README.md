# ThoughtLM Minimal

A minimal PyTorch prototype for an autoregressive decoder that can "think" by emitting continuous vectors to an external memory and reading them back via cross-attention at any time.

Highlights
- Byte-level tokenizer (no external deps).
- Small decoder-only Transformer backbone.
- Thought head (R^H -> R^D) and write gate (sigmoid) at each timestep.
- Memory cross-attention applied before the LM head to condition token prediction on accumulated thoughts.
- Training driven by a YAML config file (no CLI hyperparams).

Quick start
1) Create and edit `configs/default.yaml` as needed.
2) Train:
   - Default config: `python -m thought_lm.train`
   - Custom config path: `python -m thought_lm.train configs/your.yaml`
3) Visualize with TensorBoard:
   - Ensure `run.enable_tb: true` in your config (default).
   - Logs will be under `run.tb_log_dir/run_name` (default: `runs/default`).
   - Start UI: `tensorboard --logdir runs`
   - Open the URL printed by TensorBoard.

What gets logged
- Scalars: loss, ce_mem, ce_nomem, r_hat (avg gate prob), budget loss, predictive loss, margin loss, lr
- Thought scalars: write_rate_hard (p>0.5), gate_entropy, mem_norm, mem_nonzero_frac, mem_effect, structural-token gate means (newline, colon, lparen)
- Histograms: p_gates distribution over tokens
- Images: thought/heatmap_p_gates (rows=samples, cols=token positions)

Metrics file (JSONL)
- All train/eval scalars are also written to a newline-delimited JSON file, overwritten each run.
- Default path: `metrics.jsonl` (config: `run.metrics_file`).
- Each line has fields like: `{ "phase": "train"|"eval", "step": N, ... }`.

Notes
- This is a minimal educational prototype, not an optimized trainer.
- Files aim to follow: ≤400 LOC per file, ≤100 LOC per function (roughly), PEP8/PEP257, and are black/flake8/mypy friendly.

Using a proper dataset (streamed)
- Set in `configs/default.yaml`:
  ```yaml
  data:
    hf_dataset:
      name: bigcode/the-stack-smol
      split: train
      text_field: content
      languages: [python]
      streaming: true
      shuffle_buffer: 10000
      max_samples: null
      filter_long: true  # drop samples whose tokenized length exceeds seq_len+1
  ```
- Install deps: `pip install -r requirements.txt` (includes `datasets`).
- The loader will stream and skip over-long samples; shorter samples are padded to fixed length.

