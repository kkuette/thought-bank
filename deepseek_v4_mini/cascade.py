"""
Cascade mémoire v3 — spec utilisateur 2026-07-12 (verrouillée le soir même) :
« débordement en 2 temps × fractale max_mem ».

Structure. Le bloc 0 est la banque vive du modèle (inchangée, dans le graphe).
Chaque niveau k >= 1 a DEUX positions ; chaque position est un tenseur borné à
max_mem tranches, une tranche = une position PLEINE du niveau k-1 :

    niveau 1 : tranche = slot évincé [B, mem_dim]        pos pleine = [B, M, D]
    niveau 2 : tranche = matrice [B, M, D]               pos pleine = [B, M, M, D]
    niveau k : pos pleine = [B, M^k, ..., D]  (ordre k+2 avec le batch)

Flux (débordement en 2 temps) :
  temps 1 — la pos 0 se REMPLIT (une tranche à chaque arrivée) pendant que la
            pos 1 reste pleine : le read du niveau ne voit jamais d'init pur ;
  temps 2 — pos 0 pleine => la pos 1 (doyenne) descend ENTIÈRE comme une
            tranche de la pos 0 du niveau k+1, pos 0 glisse en pos 1, une
            neuve s'ouvre. Sous le niveau `depth`, descendre = disparaître
            (l'éviction du fond, seule destruction).

Read (merge-at-read v1 = moyenne, validée zero-shot jusqu'à avg64) : chaque
niveau rend TOUJOURS une matrice [B, <=M, D] — la moyenne de toutes les
matrices écrites qu'il contient (pos 1 + pos 0 partielle). Au niveau 1 la
pos 0 est partielle au grain SLOT : moyenne par index de slot sur le contenu
écrit seulement (jamais de padding zéro dans la moyenne — pas d'init dilué).

Tout est stocké DÉTACHÉ (hors graphe) : le TBPTT n'atteint que le bloc 0,
le read des niveaux profonds apprend, leur write non (pattern G2).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


class CascadeMemory:
    """État par conversation. Pas un nn.Module : aucune trainable var ici."""

    def __init__(self, depth: int, max_mem: int):
        assert depth >= 1
        self.depth = depth
        self.M = max_mem
        self.lv: Dict[int, dict] = {
            k: {"p0": [], "p1": None} for k in range(1, depth + 1)
        }
        self.n_pushed = 0     # slots reçus (stats/tests)
        self.n_destroyed = 0  # unités sorties par le fond (stats/tests)

    # ── écriture ────────────────────────────────────────────────────────────

    def push_slot(self, slot: torch.Tensor) -> None:
        """Reçoit le slot que la banque vive vient d'évincer. [B, D]."""
        self.n_pushed += 1
        self._push(1, slot.detach())

    def _push(self, k: int, unit: torch.Tensor) -> None:
        if k > self.depth:
            self.n_destroyed += 1
            return
        L = self.lv[k]
        L["p0"].append(unit)
        if len(L["p0"]) == self.M:
            if L["p1"] is not None:
                self._push(k + 1, L["p1"])
            L["p1"] = torch.stack(L["p0"], dim=1)   # [B, M, ...unit]
            L["p0"] = []

    # ── lecture ─────────────────────────────────────────────────────────────

    def read(self, k: int) -> Optional[torch.Tensor]:
        """Matrice mergée du niveau k : [B, M, D] (ou [B, m<M, D] en tout début
        de vie, ou None si le niveau n'a encore rien reçu)."""
        L, M = self.lv[k], self.M
        if k == 1:
            p1, p0 = L["p1"], L["p0"]
            if p1 is None and not p0:
                return None
            if p1 is None:
                return torch.stack(p0, dim=1)                     # [B, m, D]
            if not p0:
                return p1
            out = p1.clone()
            part = torch.stack(p0, dim=1)                         # [B, m, D]
            m = part.size(1)
            out[:, :m] = (out[:, :m] + part) / 2.0
            return out
        mats: List[torch.Tensor] = []
        if L["p1"] is not None:
            mats.append(self._flat(L["p1"]))
        for u in L["p0"]:
            mats.append(self._flat(u))
        if not mats:
            return None
        return torch.cat(mats, dim=1).mean(dim=1)                 # [B, M, D]

    @staticmethod
    def _flat(u: torch.Tensor) -> torch.Tensor:
        """[B, ..., M, D] → [B, n_mat, M, D] : toutes les matrices contenues."""
        B, D = u.size(0), u.size(-1)
        M = u.size(-2)
        return u.reshape(B, -1, M, D)

    # ── banques par couche pour le forward ──────────────────────────────────

    def layer_banks(
        self, live_bank: torch.Tensor, layer_map: List[int]
    ) -> List[Optional[torch.Tensor]]:
        """layer_map[i] = niveau lu par la couche i (0 = banque vive).
        Les niveaux > depth sont rabattus sur depth (flag de profondeur)."""
        cache: Dict[int, Optional[torch.Tensor]] = {0: live_bank}
        out: List[Optional[torch.Tensor]] = []
        for lvl in layer_map:
            lvl = min(lvl, self.depth)
            if lvl not in cache:
                r = self.read(lvl)
                cache[lvl] = None if r is None else r.to(live_bank.dtype)
            out.append(cache[lvl])
        return out

    # ── stats (probes / logs) ────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        s = {"pushed": self.n_pushed, "destroyed": self.n_destroyed}
        for k, L in self.lv.items():
            s[f"lv{k}_p0"] = len(L["p0"])
            s[f"lv{k}_p1"] = 0 if L["p1"] is None else 1
        return s
