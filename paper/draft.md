# A Trained Fast-Weight Memory: Continual Rule Binding at Inference Without Backward

*Draft v0.1 — 2026-07-06. Markdown master; LaTeX port after prose freeze.*

## Abstract

Continual learning at inference usually means test-time training (TTT):
gradient steps on a clone of the model. We study the alternative the
fast-weight literature has long promised: a small bank of vectors, written by
the forward pass itself, that modulates subsequent computation — no backward,
no optimizer, no weight copy. On a keyed multi-turn rule task (a fresh modular
rule per conversation, unseen queries, K=2 concurrent rules), a 3M-parameter
transformer with an 8-slot bank learns to (i) install a never-trained rule
from a single 13-token presentation (held-rule accuracy 0.79–1.00 across two
seeds; chance 0.008), (ii) retain it across turns and slot eviction, and
(iii) replace it mid-conversation in one forward pass (post-switch accuracy
0.95 train / 0.78 held; old-rule persistence 0.000). Head-to-head on the same
conversations, TTT with a full learning-rate sweep memorizes its adaptation
examples (pair accuracy 0.99) yet transfers exactly nothing to unseen
queries, while costing 138× more per rule update and destroying 62% of the
untouched concurrent rule; in-window ICL is also at chance — the bank is the
model's only functional adaptation pathway. Crucially, none of this is
emergent from the architecture: the same architecture trained without
mid-conversation rule switches perseverates completely (old-rule persistence
1.000 zero-shot). Memory *policy* — what to keep, when to overwrite, how to
write on a dirty bank — is a trained behaviour, installed by randomizing
conversation structure at training time. We map the boundary (an untrained
rule family defeats bank and TTT equally: the limit is the meta-learned
envelope, not the mechanism) and report a seed-level bifurcation between
selective-update and flush-and-rewrite replacement policies. Training the
read/write circuit requires breaking an ignore-the-bank fixed point; we give
the recipe (teacher-forced code bootstrap with annealed blending,
mastery-gated rule curriculum) and the diversity threshold below which the
read memorizes instead of generalizing.

## 1. Introduction

A model deployed in a conversation, a coding session, or a stream of events
must absorb new bindings — this user's constraint, this variable's meaning,
this rule as of now — and use them many steps later. Current practice offers
two routes. Keep the information as *data*: in the context window, a
KV-cache, or a retrieval store, and pay attention over it forever. Or push
it into *weights* by test-time training (TTT): clone the model, take
gradient steps on the new examples, and pay a backward pass, an optimizer,
and the well-known price of sequential gradient updates — catastrophic
interference. The fast-weight literature has long promised a third route:
a small memory, written by the forward pass itself, that modulates the
computation as *weights* — no backward, no clone, no growing window. What
has been missing is a controlled demonstration that such a memory can be
*trained to work*: to bind genuinely novel content at inference,
generalize it to unseen inputs, retain it under capacity pressure, and
overwrite it on demand.

This paper provides that demonstration at small scale, with the controls
that scale would blur. A 3M-parameter transformer carries an 8-slot bank
of 32-dimensional vectors; a write head appends one vector per segment,
and a hypernetwork expands each slot into a low-rank MLP layer applied to
the token stream — the bank is read as fast weights, never attended as
data. On a keyed rule task where the binding crosses turns *only* through
the bank, the trained model installs a never-trained rule from a single
13-token presentation (0.79–1.00 on unseen queries across two seeds;
chance 0.008), retains it beyond the physical eviction of its slot, and
replaces it mid-conversation in one forward pass, evacuating the old rule
completely.

The controlled comparison is the first contribution. On the same
conversations and the same checkpoint, test-time training with a full
learning-rate sweep converges on its adaptation examples (pair accuracy
0.99) and transfers *exactly nothing* to unseen queries of the same rule;
in-window ICL is at chance even on trained rules. The bank is not merely
cheaper than the alternatives — 138× cheaper per rule update, with −14%
collateral on a concurrent rule where sequential TTT costs −62% — it is
the model's *only* functional adaptation pathway (§7).

The second contribution is, we believe, the more general one: **memory
policy is a trained behaviour, not an architectural property.** The same
architecture, trained to the same held-rule competence but on conversations
of fixed structure, perseverates totally when a rule changes zero-shot
(old-rule persistence 1.000): its write head cannot even produce a
readable code on a non-empty bank. Randomizing conversation *structure*
during training — lengths, switch positions, switch counts — installs the
full policy: persistence 0.000 at every switch position, clean dirty-bank
writes, no retention cliff (§8). The architecture supplies a substrate;
the training distribution decides what the memory *does*, the way
augmentation decides a vision model's invariances. Anyone equipping a
model with a memory mechanism should expect the same: an inference-time
behaviour the training distribution never exercised should be presumed
absent until probed.

Third, we show the training problem is itself non-trivial and give a
working recipe. Joint training of write, storage and read collapses into
an ignore-the-bank fixed point; breaking it requires a teacher-forced code
bootstrap (annealed blending plus distillation — each alone fails), a
mastery-gated curriculum, and above all rule *diversity*: below a
threshold the read memorizes its repertoire and scores exactly zero on
held rules (§5).

We state the boundary as precisely as the claim. Held rules are fresh
*parameter bindings* within a meta-learned family, not new laws: on a
never-trained family (subtraction on the same circle), the bank, TTT, and
ICL all fall to chance — the frontier is the meta-training envelope, not
the adaptation mechanism (§7). And honesty about replication: across two
seeds every headline number replicates except the *selectivity* of
replacement, which bifurcates between a selective-update and a
flush-and-rewrite attractor (§9); we report both.

## 2. Related work

**Fast weights.** The idea that one network's activity should program
another's weights goes back to Schmidhuber (1992) and was revived for
attention-era models by Ba et al. (2016). Schlag et al. (2021) showed
linear attention *is* a fast-weight programmer, and the modern recurrent
line (DeltaNet-style delta rules, Titans and successors) trains
token-granular write rules inside the sequence mixer, at scale, for
language modelling perplexity. Our bank differs in granularity and in
question: one write per *segment*, read as a low-rank MLP over a
persistent multi-slot state, and evaluated not by perplexity but by an
isolable behavioural claim — can a fresh binding be installed, retained,
and replaced at inference, with the bank as the only route? The
memory-as-weights reading (slot → generated layer) also separates us from
memory-augmented models that attend to stored vectors as data.

**Memory-augmented networks.** NTM and DNC trained differentiable
read/write policies end-to-end and are the closest ancestors of our
trained-policy claim; modern Hopfield layers store patterns for
associative retrieval. Our contribution to this line is the *dissociation*:
identical architecture, competence matched, and the policy (retention
horizon, overwrite discipline, dirty-bank writes) swings from absent to
complete with the training distribution of conversation structure alone —
plus the zero-shot diagnosis of what an untrained policy looks like
(total perseveration, unreadable dirty-bank writes).

**Test-time training and adaptation.** TTT as an architectural principle
(Sun et al., 2024) compiles the gradient update *into* the layer; TTT as a
practice fine-tunes a clone on test-instance data, recently with striking
results on ARC. Our TTT arm is the practice form, taken seriously as the
natural baseline for inference-time binding, with a convergence
diagnostic that separates optimization failure from basin failure: it
fits its adaptation set perfectly and transfers nothing, and under
sequential updates it exhibits classic catastrophic interference
(McCloskey & Cohen, 1989) — which our forward-only write structurally
avoids, degrading instead by eviction, a capacity knob.

**Base architecture.** The trunk miniaturizes the DeepSeek line of
efficient transformers — latent-compressed attention (MLA) and native
sparse attention, fine-grained mixture-of-experts with shared experts
(DeepSeekMoE), as consolidated in DeepSeek-V3/V4 — plus hyper-connection
residual streams. These choices are load-bearing for realism (the bank is
grafted onto a production-style stack, not a bespoke toy), not for the
claims.

**Meta-learning and its envelope.** Our setting is meta-learning in the
MAML/in-context sense: training installs a family; inference binds its
parameters. The out-of-family boundary we measure (subtraction defeats
every arm) is the small-model, controlled version of the
counterfactual-task literature on LLMs, where out-of-distribution variants
of trained competences degrade sharply but remain above chance because a
web-scale envelope contains composition primitives our 3M-parameter
envelope lacks. The diversity threshold of §5 likewise mirrors the known
transition from memorization to in-context generalization as task
diversity grows, and our bootstrap plateaus have the grokking phenomenology
(Power et al., 2022) — we treat these classical twins as diagnostic tools
throughout.

*(Citations to be completed at LaTeX port; names and years above are the
anchor set.)*

## 3. Architecture: the Thought Bank

The model is a 4-block decoder-only transformer (d_model 128, 3.08M
parameters) augmented with a *thought bank*: a FIFO buffer of M=8 slots of
dimension 32, carried across segments of a conversation as persistent state.

**Trunk.** The trunk is a miniature of the DeepSeek-V4 stack: compressed
sparse attention (a two-tier local/compressed scheme in the spirit of
DeepSeek's sparse attention, with latent-compressed queries), a
fine-grained DeepSeekMoE feed-forward with a shared expert, and
hyper-connection residual streams. We adopt this stack because it is
representative of current production architectures — the point of the
paper is what a *memory* adds to such a trunk, not the trunk itself —
and none of our claims depend on it: the bank interfaces with the trunk
only through a pooling write head and a residual read delta, both
trunk-agnostic. Three components interact with the bank.

**Write.** At the end of each segment, a write head mean-pools the final
hidden states and projects them to a single 32-dimensional vector, which is
appended to the bank; when the bank exceeds M slots the oldest is evicted.
One segment, one write — the write head has no per-token addressing and no
learned gate in the main recipe (§App. C audits the gate: it accelerates the
bootstrap by ~30% but slows post-bootstrap consolidation 4–6×; all headline
results use gate OFF).

**Read (fast weights).** The bank is read as *weights*, not as data. Each
slot vector m_i is expanded by a learned hypernetwork into a low-rank MLP
layer (A_i ∈ R^{r×d}, B_i ∈ R^{d×r}, r=16, SwiGLU gating), and the token
stream is passed through the slots sequentially:

  y ← y + B_i · σ(A_i y),  i = 1…M;  h ← h + W_o (y − y_0).

The read is a residual delta, so an empty or irrelevant bank is a near
no-op. The non-linearity is essential: summing rank-r linear maps collapses
to a single linear map, and a rank-1 outer-product read (the classical
fast-weight form) cannot express the input-dependent *mapping* our task
requires — an ablation line, not a hypothetical: the outer-product read was
our first design and never exceeded chance on rule application. The read is
applied at block 0 only; grafting reads onto all four blocks helps a larger
variant of the task (App. D) but is not needed here.

**Credit assignment.** Gradients flow into the write head only through the
read of *later* segments, so training back-propagates through the bank
across segment boundaries (truncated BPTT; the window must cover the
conversation — with a window of 1 the write head receives no gradient at
all and the bank never trains).

**Cost model.** Installing or updating one rule = one forward pass over a
13-token segment. With the 2·P·tokens FLOPs proxy (P=3.08M) that is 80
MFLOPs; we use the same proxy (×3 for forward+backward) for the TTT arm.

## 4. Benchmark: keyed fresh-rule conversations

We want the smallest task where *inference-time* memory is both necessary
and measurable: information presented once, used many turns later, never
resolvable from the current window.

**Task.** A conversation binds K=2 key tokens to rules drawn from the
family y = (x + s) mod 128, one fresh offset s per key per conversation.
It opens with one *presentation segment* per key — [key_k, x_0, y_0, …,
x_5, y_5], 13 tokens, six example pairs — followed by 8–16 *query turns*
[key_k, x_q] with x_q drawn from the symbols *not* shown in the
presentation. Each segment is processed in its own window: the only path
from a presentation to a query is through the bank. Bank ablation is
therefore an exact control, and sits at chance (1/128 ≈ 0.008) in every
experiment below.

**Fresh bindings, not fresh laws.** The train/held split is on the offset
s: multiples of 8 (15 rules) are *held out* — never seen in training —
and the remaining 112 form the training pool. A held rule at test time is
a genuinely never-trained (key → s) binding, but it lives inside the
family the model was meta-trained on. Throughout the paper we accordingly
claim *fresh parameter binding within a meta-learned family*, not the
learning of new laws; §7 measures what happens outside the family, and
the answer is the boundary of the claim.

**Structure randomization (the policy trainer).** In the policy-training
configuration, conversation *structure* is itself randomized: length is
drawn uniformly from 8–16 turns, and up to two *switches* occur at random
positions — a random key is re-presented with a new rule, on the carried
(dirty) bank, and subsequent queries for that key follow the new rule.
A conversation can thus produce up to 20 writes into 8 slots, so retention
must survive eviction. The optimizer steps once per conversation. §8 shows
that this distribution, and nothing architectural, is what installs the
memory policy.

## 5. Training the memory: breaking the ignore-bank fixed point

Joint training of write, storage and read fails from scratch: the read
initially extracts nothing, so the loss gradient prefers routing around the
bank; the write head then receives no useful signal, and the system settles
into an *ignore-the-bank* fixed point (CE at the ln 128 floor for the
unresolvable queries, bank ablation gap ≈ 0). The failure is not in the
read itself: given any *fixed* consistent code for s injected into the
bank — one-hot, random frozen, learned embedding — the read learns to apply
it almost perfectly in isolation. The wall is credit assignment through the
write→store→read loop. Three ingredients break it:

1. **Teacher-forced code bootstrap.** During early training the bank
   content for a presentation is blended toward a fixed *Fourier code* of
   the rule (harmonics k ≤ 8 of s on the circle Z_128):
   slot ← β·teacher + (1−β)·write, with β annealed 1 → 0 over 300 steps
   once the curriculum (below) reaches 64 rules. A cosine distillation loss
   pulls the write head toward the teacher during the blend. The teacher is
   a *kick*, not a target the model keeps: after the anneal the write drifts
   off the Fourier circle into its own anisotropic code map (App. B), and
   distillation never fully converges — by design it only has to hold the
   loop together until the read has something consistent to learn from.

2. **Mastery-gated curriculum.** The rule pool starts at 16 offsets and
   doubles (16→32→64→112) each time a CE-mastery criterion is met, with a
   minimum dwell. Fixed-schedule versions of the same curriculum fail:
   either the early pool is too small for too long (the read memorizes) or
   the anneal fires before the loop is closed (the code collapses).

3. **Blend + distillation jointly.** β-blending without the distillation
   loss is at chance; distillation without blending never closes the loop.
   Neither half-measure works — the pair is the active ingredient.

**Diversity threshold.** With ≤25 training rules the read *memorizes*: train
accuracy reaches 0.99 while held rules score exactly 0.000 — the read snaps
any held code to its nearest trained neighbour, and sharpening on the
repertoire is orthogonal to interpolation on the manifold. At 112 rules,
held tracks train throughout. The transition sits somewhere in (25, 112];
we did not bracket it further. Diversity of *rules* is what converts the
read from a lookup into a function.

**What survives randomized structure.** The full recipe transfers to the
structure-randomized distribution of §4 at a cost: curriculum milestones
arrive ~2× later (pool doublings at steps 553/814/1004 vs 259/437/592 on
fixed structure), the anneal completes (distill 0.027), and held ≥ train
mid-run. Pushing randomization further — K drawn from 2–5 *and* 8–32 turns
*and* unconstrained switches — never bootstraps at this model size: CE
stays at floor and the distillation loss *rises*, the signature of the
fixed point re-forming. The structural-entropy wall is real; we bracketed
it rather than crossed it.

## 6. Results I: a functional, generalizing memory

All numbers in this section are computed on conversations with *unseen
query symbols*; "held" additionally means the rule offset s was never seen
in training. Chance is 0.008. The policy-trained cell was run twice
(seeds 42 and 43, identical config and data stream).

| capability | seed 42 (@3000) | seed 43 (@4000) |
|---|---|---|
| train rules, unseen queries | 0.951–0.987 | ~1.000 |
| **held rules (never trained)** | **0.792–0.828** | **0.997–1.000** |
| replacement: post-switch accuracy (train / held target) | 0.953 / 0.777 | 0.91–1.00 |
| old-rule persistence after switch (STICK), positions 2–14 | 0.000 | 0.000–0.051 |
| dirty-bank write identifiability (1-NN vs clean-code dictionary) | 0.90 | 0.95 |
| bank ablated | 0.008 (chance) | chance |

**Binding.** A single 13-token presentation of a never-trained rule yields
0.79–1.00 accuracy on unseen queries across the two seeds. The band is the
central claim: the bank is not a cache of trained associations but a
*generalizing* memory — the write places a fresh binding somewhere the read
can use, for offsets the read has never been supervised on.

**Retention.** There is no cliff at the bank's capacity. In a 16-turn
conversation with a switch at turn 8, the first rule's code is physically
evicted from all 8 slots (1-NN identification drops to ~0), yet post-switch
accuracy is 0.924. Retention outlives the slot that carried it.

**Mechanism: redundant superposition (Fig. 4).** Probes on both trained
seeds show the 8 slots converge to nearly the same vector: the *bank's*
effective rank at the end of a conversation is 1.13 / 1.48 out of 8
(seeds 42 / 43), with every slot-pair cosine above 0.9 (the residual
structure is a parity checkerboard — the two keys' rehearsal streams
alternate). This is not representational collapse — the ablation gap is
+4.6 nats, and across rules the clean write codes span an effective rank
of 7.3 / 12.4 of 32 dimensions once the shared mean direction is removed
(near-zero mean inter-rule cosine after centring; 1-NN identifiability
0.90–0.95). The bank stores the conversation's bindings as one superposed
vector, *copied* into every slot; the key-conditioned read disambiguates
at application time. The redundancy is what buys eviction robustness:
evicting any slot removes a copy, not the content. Two practical
corollaries: (a) low bank rank alone is not a pathology signal — it must
be read jointly with the ablation gap; (b) a switch write is genuinely
novel content — its redundancy with the resident bank drops to +0.50
(seed 42) or −0.10 (seed 43), vs ~1.0 for steady-state rehearsal writes —
and the bank re-converges to the new superposition within one turn
(redundancy back to ≥0.95 at the next write). The seed gap in that dip is
not noise; it is the §9 bifurcation, visible at the write itself.

**Replacement.** Mid-conversation, re-presenting a key with a new rule —
one 13-token forward on the dirty bank — installs the new binding at
0.953 (train) / 0.777 (held) and evacuates the old one: the model answers
the *old* rule on 0.0% of post-switch queries (STICK, swept over switch
positions 2–14). The dirty-bank write is clean: its 1-NN identification
against a dictionary of fresh-bank codes is 0.90–0.95, indistinguishable
from writes on an empty bank.

## 7. Results II: head-to-head against test-time training

The natural objection to §6 is that gradient-based adaptation would do the
same, better. We test it on the *same 64 conversations, same checkpoint*,
four arms (Table 3):

- **bank** (ours): presentations written to the bank, forward only.
- **TTT**: bank ablated; a per-conversation clone of the full model takes
  AdamW steps on the conversation's 12 example pairs formatted exactly as
  queries ([key, x] → y); learning rate swept over {3e-4, 1e-3, 3e-3},
  evaluated at 1–50 steps; best cell reported.
- **ICL**: the example pairs placed *in the query window* — the standard
  in-context route.
- **ablate**: no presentation at all (chance floor).

| pool | bank | TTT (best over LR × steps) | ICL in-window | ablate |
|---|---|---|---|---|
| train | **0.992** | 0.008 | 0.006 | 0.006 |
| held | **0.799** | 0.002 | 0.010 | 0.010 |
| subtraction (fresh family) | 0.012 | 0.004 | 0.002 | 0.008 |

**The TTT arm is healthy — that is the point.** Its pair loss falls from
5.12 to 0.03 and it reaches 0.99 accuracy *on its own adaptation
examples*: the optimization converges, and more steps change nothing.
It memorizes the 12 pairs and transfers exactly nothing to unseen queries
of the same rule. Fifty gradient steps land in a lookup-table basin; the
one-forward write lands in the generalizing one. The failure is the basin,
not the budget.

**The bank is the only functional pathway.** ICL is at chance *even on
trained rules* — surprising, since the presentation-turn format is
supervised during training — meaning this model simply has no in-window
adaptation route; and gradient adaptation, as above, fits without
transferring. Whatever competence the trunk has for the rule family is
reachable only through the bank code.

**The boundary, measured in the same protocol.** The third row is the
fairness arm: rules become y = (x − s) mod 128 — the same circle, the same
geometry, the reversed direction, never trained. Here TTT is *allowed* to
leave the meta-learned family (it updates weights); the bank is not
(forward only). Both are at chance. The boundary of inference-time
adaptation in this regime is the meta-training envelope, not the
forward-vs-gradient distinction. (A gate-ON control checkpoint, whose
write codes live off the Fourier circle entirely, is also at chance on
subtraction: no code geometry we trained buys family transfer.)

**Replacement under a concurrent load (Table 4).** Act two takes the TTT
protocol seriously as a *continual* learner. Both arms first install both
keys' rules (bank: two forwards; TTT: 50 steps to pair-fit 1.000). Then
key 0's rule is replaced while key 1 must keep serving. Since TTT's query
accuracy is chance regardless, we grant it the most favourable metric
available: retention of its own pair-fit on the untouched key.

| | bank | sequential TTT (50-step update) |
|---|---|---|
| update cost | **80 MFLOPs** (one 13-token forward) | 11,075 MFLOPs = **138×** |
| new rule, unseen queries | **0.953 train / 0.777 held** | chance (pair-fit 0.97) |
| untouched key | 0.977 → 0.832–0.844 | pair-fit 1.000 → 0.383–0.344 (**−62%**) |
| old rule | evacuated (STICK 0.000) | 0.979 → 0.078 |

The two collateral losses are qualitatively different. The bank's −0.14 on
the untouched key is *eviction pressure* — the switch adds writes to an
8-slot buffer — a capacity knob, adjustable by M. TTT's −62% is
catastrophic interference, intrinsic to sequential gradient updates on
shared weights. One mechanism degrades by forgetting copies; the other by
overwriting the function.

## 8. Results III: memory policy is a trained behaviour

Everything in §6 could be read as a property of the architecture. It is
not. We ran the identical switch probe, zero-shot, on a checkpoint of the
*same architecture* trained to the same held-rule competence (0.85) but on
*fixed* structure — 8 turns, no mid-conversation switches:

| | fixed-structure training (zero-shot switch) | structure-randomized training |
|---|---|---|
| STICK (old-rule persistence) | **1.000** | **0.000** (positions 2–14) |
| post-switch new-rule accuracy | 0.000 | 0.74–0.95 |
| dirty-bank write identifiability (1-NN) | 0.05–0.10 | 0.90 |
| old code physically evacuated | no | yes |
| K=2 key routing across the switch | intact (~1.00) | intact |

The zero-shot model *perseverates totally*: presented with a new rule
mid-conversation, it keeps applying the old one to every subsequent query.
The mechanism is visible at the write: on a dirty bank its write head
deposits a code that matches nothing in the clean-code dictionary (1-NN
0.05) — writing on a non-empty bank is simply out of its training
distribution. Only key routing survives zero-shot. Retention shows the
same signature: models trained on a fixed 8-turn horizon exhibit an exact
FIFO cliff at the training horizon, while the structure-randomized model
shows none (§6).

The delta between the two columns is *only* the training distribution of
conversation structure — same architecture, same bank, same rule family,
same recipe. What the memory *does* — keep, overwrite, write-on-dirty,
survive eviction — is decided by the distribution, the way visual
invariances are decided by augmentation. The architecture is a substrate;
the policy is data. We are careful not to overstate this: key routing did
survive the zero-shot probe, and the trained policy generalizes within
its envelope (switch positions are interchangeable across the trained
range; held rules install mid-conversation). But every policy that was
absent from the training distribution was absent from the behaviour —
down to the write head being unable to produce a readable code on a bank
state it had never written on. We believe this is the observation most
likely to transfer beyond our toy: a memory mechanism does not come with
its policy included, and each inference-time behaviour one wants
(retention horizons, overwrite discipline — or usages we did not test,
such as interleaved ingestion and later recall of multiple contexts)
should be presumed to need matching pressure in the training
distribution until probed.

## 9. Findings and boundaries

**A seed-level bifurcation in replacement policy.** The two seeds of the
policy cell replicate every §6 number except one. When key 0 is switched,
seed 42 *preserves* key 1 (categorized accuracy on the true training
stream: 0.863 on the non-switched key post-switch) while seed 43 flushes
the whole bank and rewrites only the switched binding (0.011 — chance) —
despite being the stronger model on every other axis (held 0.997–1.000).
The non-switched key accounts for 17% of query tokens, so seed 43 pays
chance-level loss on 17% of its queries for the entire run without
escaping the basin: final CEs are indistinguishable. The attractor is
visible at the write itself: the switch write's redundancy with the
resident bank is +0.50 for seed 42 (it preserves the resident
superposition) but −0.10 for seed 43 (it displaces it; Fig. 4C). Two
attractors of the
update policy — *selective component update* vs *flush-and-rewrite* — are
decided during the bootstrap and gradient does not cross between them.
Selective replacement plausibly requires a copy-forward circuit (the write
re-reads the bank and re-emits the untouched component); average loss
pressure evidently does not force it. We report both attractors rather
than selecting the flattering seed; steering the basin (event ordering
during bootstrap, or introducing switches only after retention
consolidates) is future work.

**Out-of-family is out of reach for every arm** (§7, subtraction row) —
and this is consistent with sequential and mixed two-family training
attempts at this scale, all negative, and with the counterfactual-task
literature on large models, where out-of-family performance degrades
sharply but stays above chance precisely because a large pretraining
envelope contains the composition primitives our 3M-parameter envelope
lacks.

**Composition is a policy gap, not a capacity gap.** The trunk can chain:
feeding f(x)'s output back as a new query (external chaining) scores
0.961. Asked to compose internally — f(f(x)) in one query — the model is
at chance. Every piece of an iterative computation through the bank
exists; the *behaviour* of chaining is absent, exactly as replacement was
absent before §8. We conjecture it is trainable by the same method
(structure pressure), which would make the bank a latent scratchpad — a
fast-weight analogue of chain-of-thought — and leave it to future work.

## 10. Limitations

- **Scale and domain.** 3.08M parameters, one synthetic family (modular
  addition, S=128, K=2). A multiplicative family (y = 3x+s) does not
  install at this model size with the same recipe. No natural-language
  results; a synthetic gist-carrying benchmark is positive but
  preliminary.
- **TTT baseline strength.** Our TTT arm is full-parameter AdamW with an
  LR sweep and a convergence diagnostic. Parameter-efficient TTT (LoRA),
  larger adaptation sets, or regularized updates could shift the margins
  of Tables 3–4, though the memorize-without-transfer signature suggests
  the failure is not budget-limited.
- **Teacher specificity.** The bootstrap teacher is a Fourier code — the
  natural geometry for this family. The recipe's generality beyond
  circular structure is untested.
- **Seeds.** The central claim is replicated across two seeds; the
  selectivity bifurcation would need a proper basin census (which our
  compute budget did not allow) before any claim about attractor
  prevalence.
- **Wording.** Held rules are fresh *bindings* within the meta-learned
  family, not new laws; we have been careful to claim no more.

## 11. Reproducibility

Code, configurations, and analysis probes are available at
github.com/kkuette/thought-bank. The policy cell trains in ≈5 h on a
single RTX 3090; every probe and both TTT arms run on CPU. Seeds 42/43;
data generators are deterministic given the config; checkpoints saved
every 100 steps. An end-to-end script reproducing Tables 2–4 and Figure 5
from a fresh clone is provided (repro/).

## Appendices (planned)

- **A. Training dynamics** — full curves for the policy cell (CE, distill,
  curriculum milestones, anneal window, WSD decay), seed 42 vs 43.
- **B. Post-anneal code drift** — the write leaves the Fourier circle;
  ridge probes die, 1-NN identification survives; why distillation
  non-convergence is health, not failure.
- **C. Write-gate audit** — gate ON accelerates curriculum milestones
  ~30% but slows post-anneal consolidation 4–6×; off-circle code map with
  rule collisions; why the headline recipe is gate OFF.
- **D. Read depth** — grafting fast-weight reads onto all 4 blocks
  (LoRA-B-style zero-init) on the larger S=256 variant.
- **E. Negative results** — full-randomization bootstrap failure (dsv4v);
  fixed-schedule curricula; β-blend without distill; outer-product read;
  two-family training attempts.
