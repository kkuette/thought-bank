"""B4 — canal DeltaNet interne (backlog 2026-07-13 ; steelman linear-attention
à 97M, science interne — l'engagement public r/LocalLLaMA reste « à l'échelle
cible dans la run financée »).

Question : à MODÈLE constant, un canal gated delta-rule inter-tours porte-t-il
le même GAP que le carry de banque ? Le modèle (read fast-weight, write
intra-chunk, tout model.py) est STRICTEMENT inchangé — seul le canal
INTER-CHUNKS change :

  bras banque : bank_{i+1} = o["mem_bank"] (write head entraîné, slots FIFO)
  bras delta  : S_{i+1} = gated-delta-rule(S_i, embed(chunk_i)) ; le modèle
                consomme S reshapé en pseudo-banque [max_mem, mem_dim]
                (RMS-norm par slot + gain appris, pour matcher l'échelle
                que le read attend), o["mem_bank"] est ignoré.

Gated delta rule (Yang et al. 2024), séquentielle sur les tokens du chunk :
    S_t = α_t (S_{t-1} − β_t k_t k_tᵀ S_{t-1}) + β_t k_t v_tᵀ
avec k normalisé, β = force d'écriture, α = porte d'oubli (init lente).
État [d_k × d_v] avec d_k·d_v = max_mem·mem_dim (reshape exact). Capacité
= rang ≤ d_k (64) vs 8 slots × 512 dims écrits par une tête entraînée —
les deux issues sont informatives (cf. dsv4mini-baseline-deltanet-steelman).

L'update tourne en float32 hors autocast (512 pas séquentiels en bf16
dérivent) ; ~50k params (W_k/W_v 384×64), ajoutés à l'optimiseur du trainer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaChannel(nn.Module):
    def __init__(self, d_model, max_mem, mem_dim, d_k=64):
        super().__init__()
        assert (max_mem * mem_dim) % d_k == 0, "d_k doit diviser max_mem*mem_dim"
        self.d_k, self.d_v = d_k, (max_mem * mem_dim) // d_k
        self.M, self.D = max_mem, mem_dim
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v, bias=False)
        self.w_beta = nn.Linear(d_model, 1)     # force d'écriture par token
        self.w_alpha = nn.Linear(d_model, 1)    # porte d'oubli par token
        self.gain = nn.Parameter(torch.ones(()))
        nn.init.constant_(self.w_alpha.bias, 4.0)   # σ≈0.982 : oubli lent à l'init
        nn.init.constant_(self.w_beta.bias, -2.0)   # σ≈0.12 : écriture douce à l'init

    def init_state(self, B, device):
        return torch.zeros(B, self.d_k, self.d_v, device=device)  # float32

    def update(self, S, emb):
        """emb [B, L, d_model] (embeddings du modèle) -> nouvel état [B, dk, dv]."""
        emb = emb.float()
        k = F.normalize(self.W_k(emb), dim=-1)          # [B, L, dk]
        v = self.W_v(emb)                               # [B, L, dv]
        beta = torch.sigmoid(self.w_beta(emb))          # [B, L, 1]
        alpha = torch.sigmoid(self.w_alpha(emb))        # [B, L, 1]
        for t_ in range(emb.size(1)):
            kt = k[:, t_].unsqueeze(-1)                 # [B, dk, 1]
            vt = v[:, t_].unsqueeze(1)                  # [B, 1, dv]
            bt = beta[:, t_].unsqueeze(-1)              # [B, 1, 1]
            at = alpha[:, t_].unsqueeze(-1)
            S = at * (S - bt * kt @ (kt.transpose(1, 2) @ S)) + bt * kt @ vt
        return S

    def to_bank(self, S, dtype):
        """État -> pseudo-banque [B, max_mem, mem_dim] à l'échelle du read."""
        b = S.reshape(S.size(0), self.M, self.D)
        b = b / b.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return (self.gain * b).to(dtype)
