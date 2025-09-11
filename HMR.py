!pip install "transformers==4.41.2" "tokenizers==0.19.1" "huggingface_hub>=0.23.0"
# Restart runtime after this install
import os, shutil, pathlib
# Clear any cached copy that might be corrupted/incompatible
cache_dirs = [
    os.path.expanduser("~/.cache/huggingface"),
    "/root/.cache/huggingface",
]
for d in cache_dirs:
    if os.path.isdir(d):
        print("Clearing cache:", d)
        shutil.rmtree(d, ignore_errors=True)

import os
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"


import os, math, json, random, io, datetime
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from datasets import load_dataset
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from huggingface_hub import HfApi, HfFolder, hf_hub_download
from google.colab import userdata

# ----------------------------
# Config
# ----------------------------
UPDATE_README = False
HF_REPO_ID = "SofiTesfay2010/HRM-LLM"
LOCAL_CHECKPOINT_PATH = "local_training_state.pt"
LOCAL_WEIGHTS_PATH = "pytorch_model.bin"
SEED = 42
MIXED_PRECISION = False  # start False for stability; turn True later
GRAD_ACCUM_STEPS = 1
NUM_EPOCHS = 5
BLOCK_SIZE = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WARMUP_STEPS_RATIO = 0.1
MAX_HALT_STEPS = 8
PONDER_WEIGHT = 1e-2
MODEL_CONFIG = {"d_model": 512, "n_heads": 8, "d_ff": 2048, "dropout": 0.1}
T5_TOKENIZER_REPO = "t5-small"  # change to your repo with spiece.model if desired

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Hub auth
# ----------------------------
try:
    HF_TOKEN = userdata.get("HF_TOKEN")
    print("Hugging Face token found in Colab secrets.")
    HfFolder.save_token(HF_TOKEN)
    print("Login to Hugging Face Hub successful.")
except userdata.SecretNotFoundError:
    print("HF_TOKEN secret not found. Please create it in the Colab sidebar under 'Secrets'.")
    HF_TOKEN = None

# ----------------------------
# Tokenizer (slow T5 SentencePiece)
# ----------------------------
print("Loading tokenizer (T5 slow)...")
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"  # enforce slow path
tokenizer = T5Tokenizer.from_pretrained(
    T5_TOKENIZER_REPO,
    use_fast=False,
    trust_remote_code=True,
)

# Ensure special tokens (T5 has <pad> by default; eos_token is </s>)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "left"
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}; eos={tokenizer.eos_token}; pad={tokenizer.pad_token}")

# ----------------------------
# Layers and Model
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

class SwiGLUMuchPelu(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        activated = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(activated))

class HRMBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMuchPelu(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class HRMInner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        self.dropout = nn.Dropout(config["dropout"])
        self.H_module = HRMBlock(config["d_model"], config["n_heads"], config["d_ff"], config["dropout"])
        self.L_module = HRMBlock(config["d_model"], config["n_heads"], config["d_ff"], config["dropout"])
    def forward(self, z_H, z_L, attn_mask=None, key_padding_mask=None):
        z_L_input = z_L + z_H
        z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        z_H_input = z_H + z_L_new
        z_H_new = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return z_H_new, z_L_new

class HierarchicalReasoningModel_ACTV1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inner_model = HRMInner(config)
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.halt_head = nn.Sequential(nn.Linear(config["d_model"], 1), nn.Sigmoid())
        self.max_steps = config["halt_max_steps"]
        self.ponder_loss_weight = config["ponder_loss_weight"]

    def forward(self, input_ids, labels=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        z_L = self.inner_model.token_embeddings(input_ids)
        z_H = torch.zeros_like(z_L)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        mask_val = -1e4
        causal = torch.zeros((seq_len, seq_len), device=device)
        causal = causal.masked_fill(torch.triu(torch.ones_like(causal), diagonal=1).bool(), mask_val)

        halting_probs = torch.zeros((batch_size, seq_len, self.max_steps), device=device)
        remainders = torch.ones((batch_size, seq_len), device=device)

        total_z_H = 0.1 * z_L.clone()

        eps = 1e-6
        for step in range(self.max_steps):
            p_halt = self.halt_head(z_H).squeeze(-1)
            p_halt = p_halt.clamp(eps, 1 - eps)

            is_last = (step == self.max_steps - 1)
            halt_now_prob = torch.ones_like(p_halt) if is_last else p_halt

            contrib = (remainders * halt_now_prob).clamp(min=0.0, max=1.0)
            halting_probs[:, :, step] = contrib
            total_z_H = total_z_H + contrib.unsqueeze(-1) * z_H

            remainders = (remainders * (1 - p_halt)).clamp(min=eps, max=1.0)

            if torch.all(remainders < 1e-4):
                break

            z_H, z_L = self.inner_model(z_H, z_L, attn_mask=causal, key_padding_mask=key_padding_mask)

        logits = self.lm_head(total_z_H)
        loss = None
        ponder_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config["vocab_size"]), shift_labels.view(-1))
            ponder_loss = torch.mean(torch.sum(halting_probs, dim=-1))
            loss = lm_loss + self.ponder_loss_weight * ponder_loss
        return {"loss": loss, "logits": logits, "ponder_loss": ponder_loss}

# ----------------------------
# Data
# ----------------------------
print("Loading and preparing dataset...")
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    # T5 uses </s> as eos; tokenizer.eos_token should be set
    texts = [t + (tokenizer.eos_token or "") for t in examples["text"]]
    return tokenizer(texts, add_special_tokens=False)

tokenized = raw_datasets.map(
    tokenize_function, batched=True, num_proc=2,
    remove_columns=raw_datasets["train"].column_names,
)
train_text_hf = tokenized["train"]
val_text_hf = tokenized["validation"]

class LLMDataset(Dataset):
    def __init__(self, hf_dataset, block_size):
        self.block_size = block_size
        all_token_ids = [tid for doc in hf_dataset["input_ids"] for tid in doc]
        self.examples = []
        for i in range(0, len(all_token_ids) - block_size + 1, block_size):
            self.examples.append(all_token_ids[i : i + block_size])
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return torch.tensor(self.examples[i], dtype=torch.long)

train_dataset = LLMDataset(train_text_hf, BLOCK_SIZE)
val_dataset = LLMDataset(val_text_hf, BLOCK_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ----------------------------
# Model, Optimizer, Scheduler
# ----------------------------
config = {
    "vocab_size": len(tokenizer),
    "d_model": MODEL_CONFIG["d_model"],
    "n_heads": MODEL_CONFIG["n_heads"],
    "d_ff": MODEL_CONFIG["d_ff"],
    "dropout": MODEL_CONFIG["dropout"],
    "halt_max_steps": MAX_HALT_STEPS,
    "ponder_loss_weight": PONDER_WEIGHT,
}
model = HierarchicalReasoningModel_ACTV1(config).to(device)

with torch.no_grad():
    model.inner_model.token_embeddings.weight.normal_(0.0, 0.02)
    model.lm_head.weight.normal_(0.0, 0.02)
    if hasattr(model.halt_head[0], "bias") and model.halt_head[0].bias is not None:
        model.halt_head[0].bias.zero_()

decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    name = n.lower()
    if any(k in name for k in ["bias", "norm", "rmsnorm", "layernorm"]):
        no_decay.append(p)
    else:
        decay.append(p)
optimizer = AdamW(
    [{"params": decay, "weight_decay": 0.01},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=LEARNING_RATE,
)

try:
    print(f"--- Downloading latest model from '{HF_REPO_ID}'... ---")
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=LOCAL_WEIGHTS_PATH)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print("--- Successfully loaded model from the Hub. Continuing its training. ---")
except Exception as e:
    print(f"--- Could not download model from Hub. First upload or mismatch. Error: {e} ---")

start_epoch = 0
global_step = 0

if os.path.exists(LOCAL_CHECKPOINT_PATH):
    try:
        local_checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location="cpu")
        if "optimizer_state_dict" in local_checkpoint:
            optimizer.load_state_dict(local_checkpoint["optimizer_state_dict"])
        start_epoch = local_checkpoint.get("epoch", -1) + 1
        global_step = local_checkpoint.get("global_step", 0)
        print(f"--- Resuming from Epoch {start_epoch}, global_step {global_step}. ---")
    except Exception as e:
        print(f"Warning: failed to load local training state: {e}")

total_train_steps = (len(train_loader) // max(1, GRAD_ACCUM_STEPS)) * NUM_EPOCHS
warmup_steps = max(1, int(WARMUP_STEPS_RATIO * total_train_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == "cuda"))

# ----------------------------
# Training
# ----------------------------
for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Community Training Epoch {epoch}")
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress):
        input_ids = batch.to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        labels = input_ids

        with torch.amp.autocast("cuda", enabled=(MIXED_PRECISION and device.type == "cuda")):
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"]

        if loss is None or not torch.isfinite(loss):
            print("Non-finite loss, skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_to_backprop = loss / GRAD_ACCUM_STEPS

        if scaler.is_enabled():
            scaler.scale(loss_to_backprop).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                bad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
                if bad:
                    print("Non-finite grad, skipping optimizer step.")
                    scaler.reset(); optimizer.zero_grad(set_to_none=True); continue
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
        else:
            loss_to_backprop.backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                bad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
                if bad:
                    print("Non-finite grad, skipping optimizer step.")
                    optimizer.zero_grad(set_to_none=True); continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        total_loss += float(loss.detach().item())
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_loss / max(1, len(train_loader))
    print(f"\nEpoch {epoch} | Training Loss: {avg_train_loss:.4f}")

    model.eval()
    total_val_loss = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch.to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            out = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            if out["loss"] is not None and torch.isfinite(out["loss"]):
                total_val_loss += float(out["loss"].item())
    avg_val_loss = total_val_loss / max(1, len(val_loader))
    print(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), LOCAL_WEIGHTS_PATH)
    torch.save(
        {"epoch": epoch,
         "optimizer_state_dict": optimizer.state_dict(),
         "scheduler_state_dict": scheduler.state_dict(),
         "global_step": global_step,
         "config": config},
        LOCAL_CHECKPOINT_PATH,
    )

    print("\n--- Saving and Uploading contribution to the Hub... ---")
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=LOCAL_WEIGHTS_PATH,
            path_in_repo=LOCAL_WEIGHTS_PATH,
            repo_id=HF_REPO_ID, repo_type="model",
            commit_message=f"Contribution after Epoch {epoch}. Val Loss: {avg_val_loss:.4f}",
            token=HF_TOKEN
        )
        if UPDATE_README:
            card_text = f"""---
base_model: {T5_TOKENIZER_REPO}
tags:
- hrm
- act
- community-training
metrics:
- loss
---

HRM-LLM community training

Tokenizer: {T5_TOKENIZER_REPO} (slow T5 SentencePiece)
Vocab size: {len(tokenizer)}

Epoch: {epoch}
Validation Loss: {avg_val_loss:.4f}
"""
            with open("README.md", "w") as f: f.write(card_text)
            api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md",
                repo_id=HF_REPO_ID, repo_type="model",
                commit_message=f"Update card after Epoch {epoch}",
                token=HF_TOKEN
            )
        print("--- Upload successful! Thank you for contributing. ---")
    except Exception as e:
        print(f"Upload failed: {e}")

print("Training run finished.")

# ----------------------------
# Simple chat
# ----------------------------
def chat_with_model(prompt_text, model, tokenizer, max_new_tokens=60, temperature=0.7, top_k: Optional[int]=50, top_p: Optional[float]=0.95):
    model.eval()
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids, attention_mask=attention_mask)
            next_token_logits = out["logits"][:, -1, :]
            if temperature and temperature > 0:
                next_token_logits = next_token_logits / max(1e-6, temperature)
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idx = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                    mask = torch.full_like(next_token_logits, float("-inf"))
                    mask.scatter_(1, topk_idx, topk_vals)
                    next_token_logits = mask
                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    sorted_mask = cum_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    next_token_logits[0, sorted_indices[0, sorted_mask[0]]] = float("-inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

print("\n--- HRM Community Model is Ready ---")
if "model" in locals():
    try:
        print(chat_with_model("Hello, how are you?", model, tokenizer, max_new_tokens=20))
    except Exception as e:
        print(f"Generation test failed: {e}")
else:
    print("Model not initialized.")