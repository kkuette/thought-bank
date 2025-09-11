"""
Trainer callbacks for logging perplexity and thinking-length metrics.

Functions are kept short per project rules.
"""
from __future__ import annotations

from typing import Dict, Optional, Any

import math
import os
import torch
from transformers import TrainerCallback

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


class PerplexityLoggerCallback(TrainerCallback):
    """Compute and log perplexity into evaluation/train logs and TensorBoard."""

    def __init__(self) -> None:  # noqa: D401
        super().__init__()
        self._tb: Optional[Any] = None

    def _ensure_tb(self, args) -> None:
        if self._tb is not None or SummaryWriter is None:
            return
        try:
            logdir = getattr(args, "logging_dir", None) or os.path.join(getattr(args, "output_dir", "."), "runs")
            os.makedirs(logdir, exist_ok=True)
            self._tb = SummaryWriter(log_dir=logdir)
        except Exception:
            self._tb = None

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: D401
        metrics = kwargs.get("metrics", {})
        if isinstance(metrics, dict) and "eval_loss" in metrics:
            try:
                eval_ppl = float(math.exp(metrics["eval_loss"]))
                metrics["eval_ppl"] = eval_ppl
                self._ensure_tb(args)
                if self._tb is not None:
                    self._tb.add_scalar("eval/ppl", eval_ppl, global_step=state.global_step)
            except Exception:
                pass
        return control

    def on_log(self, args, state, control, **kwargs):  # noqa: D401
        logs = kwargs.get("logs", {})
        if isinstance(logs, dict) and "loss" in logs:
            try:
                train_ppl = float(math.exp(logs["loss"]))
                logs["train_ppl"] = train_ppl
                self._ensure_tb(args)
                if self._tb is not None:
                    self._tb.add_scalar("train/ppl", train_ppl, global_step=state.global_step)
            except Exception:
                pass
        return control


class ThinkingEvalCallback(TrainerCallback):
    """Run short generation on eval set and log thinking length metrics and traces.

    Computes per-eval averages:
    - eval_think_tokens: average number of <think> tokens in generated continuation
    - eval_think_ratio: fraction of generated tokens that are <think>
    - eval_think_run_len: average contiguous run length of <think> tokens

    Optionally also logs training-time thinking metrics at a fixed interval
    by running generation on a few eval samples and injecting metrics into
    the Trainer's train logs and TensorBoard.
    """

    def __init__(
        self,
        *,
        tokenizer,
        eval_dataset,
        text_field: str,
        think_token_id: int,
        sample_size: int = 8,
        gen_max_new_tokens: int = 128,
        gen_max_new_thoughts: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        train_log_interval: int = 0,
        model: Optional[Any] = None,
        log_trace_eval: bool = False,
        log_trace_train: bool = False,
        think_logit_bias: float = 0.0,
        min_thoughts_first: int = 0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.text_field = text_field
        self.think_token_id = think_token_id
        self.sample_size = sample_size
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_max_new_thoughts = gen_max_new_thoughts
        self.temperature = temperature
        self.top_k = top_k
        self.train_log_interval = train_log_interval
        self.model = model
        self.log_trace_eval = log_trace_eval
        self.log_trace_train = log_trace_train
        self._tb: Optional[Any] = None
        self.think_logit_bias = think_logit_bias
        # If we are logging traces but no schedule provided, force at least one initial think step
        self.min_thoughts_first = max(1, min_thoughts_first) if log_trace_eval else min_thoughts_first

    @staticmethod
    def _compute_think_stats(gen_seq: torch.Tensor, start: int, think_id: int) -> Dict[str, float]:
        """Compute stats on the generated suffix gen_seq[start:]."""
        suffix = gen_seq[start:]
        total = int(suffix.numel())
        if total == 0:
            return {"think_tokens": 0.0, "think_ratio": 0.0, "think_run_len": 0.0}
        think_mask = (suffix == think_id).to(torch.int32)
        think_tokens = int(think_mask.sum().item())
        # run length: count contiguous runs of ones
        runs = 0
        run_len_sum = 0
        prev = 0
        for v in think_mask.tolist():
            if v == 1:
                if prev == 0:
                    runs += 1
                run_len_sum += 1
            prev = v
        avg_run_len = (run_len_sum / runs) if runs > 0 else 0.0
        return {
            "think_tokens": float(think_tokens),
            "think_ratio": float(think_tokens / max(total, 1)),
            "think_run_len": float(avg_run_len),
        }

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: D401
        model = self.model
        eval_dataset = self.eval_dataset
        if model is None or eval_dataset is None or len(eval_dataset) == 0:
            return control

        device = model.device if hasattr(model, "device") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model_was_train = model.training
        model.eval()

        # Sample a small, deterministic slice each eval
        start_idx = int((state.global_step or 0) % max(len(eval_dataset) - 1, 1))
        end_idx = min(start_idx + self.sample_size, len(eval_dataset))
        texts = [eval_dataset[i][self.text_field] for i in range(start_idx, end_idx)]
        if not texts:
            return control

        stats = {"think_tokens": 0.0, "think_ratio": 0.0, "think_run_len": 0.0}
        samples = 0
        with torch.no_grad():
            for idx, txt in enumerate(texts):
                enc = self.tokenizer(
                    txt, return_tensors="pt", truncation=True, max_length=512
                )
                input_ids = enc["input_ids"].to(device)
                if self.log_trace_eval and hasattr(model, "generate_with_trace"):
                    gout = model.generate_with_trace(
                        input_ids,
                        think_token_id=self.think_token_id,
                        max_new_thoughts=self.gen_max_new_thoughts,
                        max_new_tokens=self.gen_max_new_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        include_think_tokens_in_output=True,
                        think_logit_bias=self.think_logit_bias,
                        min_thoughts_first=self.min_thoughts_first,
                    )
                    gen_seq = gout["tokens"][0].detach().cpu()
                    trace = gout["thought_trace"].detach().cpu()  # [steps, B, D]
                else:
                    gen_seq = model.generate(
                        input_ids,
                        think_token_id=self.think_token_id,
                        max_new_thoughts=self.gen_max_new_thoughts,
                        max_new_tokens=self.gen_max_new_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        include_think_tokens_in_output=True,
                        think_logit_bias=self.think_logit_bias,
                        min_thoughts_first=self.min_thoughts_first,
                    )[0].detach().cpu()
                    trace = None
                s = self._compute_think_stats(
                    gen_seq, start=input_ids.size(1), think_id=self.think_token_id
                )
                for k in stats:
                    stats[k] += s[k]
                # Log a small heatmap for the first sample if trace available
                if idx == 0 and self.log_trace_eval and trace is not None and trace.numel() > 0:
                    # trace -> [steps, D]
                    td = trace[:, 0, :]
                    # Normalize to [0,1] per image for visualization
                    tmin = td.min()
                    tmax = td.max()
                    img = (td - tmin) / (tmax - tmin + 1e-6)
                    img = img.unsqueeze(0)  # [1, steps, D]
                    try:
                        if SummaryWriter is not None:
                            if self._tb is None:
                                logdir = getattr(args, "logging_dir", None) or os.path.join(getattr(args, "output_dir", "."), "runs")
                                os.makedirs(logdir, exist_ok=True)
                                self._tb = SummaryWriter(log_dir=logdir)
                            self._tb.add_image("eval/thought_trace_heatmap", img, global_step=state.global_step)
                            # Also log per-step L2 norms as histogram
                            norms = torch.linalg.vector_norm(td, dim=-1)
                            self._tb.add_histogram("eval/thought_norms", norms, global_step=state.global_step)
                    except Exception:
                        pass
                samples += 1

        if model_was_train:
            model.train()

        if samples > 0:
            for k in stats:
                stats[k] = float(stats[k] / samples)
            metrics = kwargs.get("metrics", {})
            if isinstance(metrics, dict):
                metrics.update({
                    "eval_think_tokens": stats["think_tokens"],
                    "eval_think_ratio": stats["think_ratio"],
                    "eval_think_run_len": stats["think_run_len"],
                })
            # Also write to TensorBoard directly
            try:
                if SummaryWriter is not None:
                    if self._tb is None:
                        logdir = getattr(args, "logging_dir", None) or os.path.join(getattr(args, "output_dir", "."), "runs")
                        os.makedirs(logdir, exist_ok=True)
                        self._tb = SummaryWriter(log_dir=logdir)
                    self._tb.add_scalar("eval/think_tokens", stats["think_tokens"], global_step=state.global_step)
                    self._tb.add_scalar("eval/think_ratio", stats["think_ratio"], global_step=state.global_step)
                    self._tb.add_scalar("eval/think_run_len", stats["think_run_len"], global_step=state.global_step)
            except Exception:
                pass
        return control

    def on_log(self, args, state, control, **kwargs):  # noqa: D401
        """Optionally log thinking metrics during training at fixed intervals.

        Injects train_think_* metrics into the logs dict so they are emitted to
        TensorBoard alongside loss and train_ppl.
        """
        if not self.train_log_interval or self.train_log_interval <= 0:
            return control
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            return control
        if (state.global_step or 0) == 0 or (state.global_step % self.train_log_interval) != 0:
            return control
        logs = kwargs.get("logs", {})
        # Avoid heavy compute if already computed this step
        if any(k in logs for k in ("train_think_tokens", "train_think_ratio", "train_think_run_len")):
            return control

        # Perform a tiny generation pass on a few eval samples
        try:
            model = self.model
            if model is None:
                return control
            device = model.device if hasattr(model, "device") else (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
            start_idx = int((state.global_step or 0) % max(len(self.eval_dataset) - 1, 1))
            end_idx = min(start_idx + max(1, self.sample_size // 2), len(self.eval_dataset))
            texts = [self.eval_dataset[i][self.text_field] for i in range(start_idx, end_idx)]
            if not texts:
                return control
            stats = {"think_tokens": 0.0, "think_ratio": 0.0, "think_run_len": 0.0}
            samples = 0
            with torch.no_grad():
                for idx, txt in enumerate(texts):
                    enc = self.tokenizer(
                        txt, return_tensors="pt", truncation=True, max_length=256
                    )
                    input_ids = enc["input_ids"].to(device)
                    if self.log_trace_train and hasattr(model, "generate_with_trace"):
                        gout = model.generate_with_trace(
                            input_ids,
                            think_token_id=self.think_token_id,
                            max_new_thoughts=max(8, self.gen_max_new_thoughts // 4),
                            max_new_tokens=max(16, self.gen_max_new_tokens // 8),
                            temperature=self.temperature,
                            top_k=self.top_k,
                            include_think_tokens_in_output=True,
                            think_logit_bias=self.think_logit_bias,
                            min_thoughts_first=self.min_thoughts_first,
                        )
                        gen_seq = gout["tokens"][0].detach().cpu()
                        trace = gout["thought_trace"].detach().cpu()
                    else:
                        out = model.generate(
                            input_ids,
                            think_token_id=self.think_token_id,
                            max_new_thoughts=max(8, self.gen_max_new_thoughts // 4),
                            max_new_tokens=max(16, self.gen_max_new_tokens // 8),
                            temperature=self.temperature,
                            top_k=self.top_k,
                            include_think_tokens_in_output=True,
                            think_logit_bias=self.think_logit_bias,
                            min_thoughts_first=self.min_thoughts_first,
                        )
                        gen_seq = out[0].detach().cpu()
                        trace = None
                    s = self._compute_think_stats(
                        gen_seq, start=input_ids.size(1), think_id=self.think_token_id
                    )
                    for k in stats:
                        stats[k] += s[k]
                    if idx == 0 and self.log_trace_train and trace is not None and trace.numel() > 0:
                        td = trace[:, 0, :]
                        tmin = td.min(); tmax = td.max()
                        img = (td - tmin) / (tmax - tmin + 1e-6)
                        img = img.unsqueeze(0)
                        try:
                            if SummaryWriter is not None:
                                if self._tb is None:
                                    logdir = getattr(args, "logging_dir", None) or os.path.join(getattr(args, "output_dir", "."), "runs")
                                    os.makedirs(logdir, exist_ok=True)
                                    self._tb = SummaryWriter(log_dir=logdir)
                                self._tb.add_image("train/thought_trace_heatmap", img, global_step=state.global_step)
                                norms = torch.linalg.vector_norm(td, dim=-1)
                                self._tb.add_histogram("train/thought_norms", norms, global_step=state.global_step)
                        except Exception:
                            pass
                    samples += 1
            if samples > 0:
                for k in stats:
                    stats[k] = float(stats[k] / samples)
                logs["train_think_tokens"] = stats["think_tokens"]
                logs["train_think_ratio"] = stats["think_ratio"]
                logs["train_think_run_len"] = stats["think_run_len"]
                # Also write to TensorBoard directly
                try:
                    if SummaryWriter is not None:
                        if self._tb is None:
                            logdir = getattr(args, "logging_dir", None) or os.path.join(getattr(args, "output_dir", "."), "runs")
                            os.makedirs(logdir, exist_ok=True)
                            self._tb = SummaryWriter(log_dir=logdir)
                        self._tb.add_scalar("train/think_tokens", stats["think_tokens"], global_step=state.global_step)
                        self._tb.add_scalar("train/think_ratio", stats["think_ratio"], global_step=state.global_step)
                        self._tb.add_scalar("train/think_run_len", stats["think_run_len"], global_step=state.global_step)
                except Exception:
                    pass
        except Exception:
            # Be robust: never break training due to optional logging
            pass
        return control
