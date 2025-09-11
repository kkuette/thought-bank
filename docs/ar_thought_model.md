# ARThoughtModel Assembly Diagram

This diagram shows how inputs flow through the ARThoughtModel to produce both the next-token prediction and the next-thought vector, and how the thought memory is updated.

```mermaid
flowchart LR
    %% Inputs
    A[Tokens (B x T)] -->|lookup| B[Token Embedding]
    A -->|positions| C[Positional Embedding]
    B --> D[Sum]
    C --> D

    %% Backbone
    subgraph G[Autoregressive Backbone]
        D --> E[Stack of Causal Transformer Blocks]
        E --> F[h_last (B x d_model)]
    end

    %% Thought context from memory
    subgraph H[Thought Context]
        M[(Thought Memory\n(B x L x thought_dim))] --> I[Map to d_model\n(MLP)]
        F --> J[Query]
        I --> K[Multihead Attention\n(q = h_last, k,v = mem_proj)]
        J --> K
        K --> L[Context (B x d_model)]
    end

    %% Fusion
    F --> N[Residual Fusion\n(h_last + context)]
    L --> N

    %% Dual heads
    subgraph O[Dual Output Heads]
        N --> P[TokenHead\n(LayerNorm + Linear/tied weights)]
        N --> Q[ThoughtHead\n(LayerNorm + MLP)]
        P --> R[Next-token Logits\n(B x vocab_size)]
        P -. optional .-> R2[Predicted Token Embedding\n(B x d_model)]
        Q --> S[Next Thought Vector\n(B x thought_dim)]
    end

    %% Memory update
    S --> T[ThoughtMemory.push\n(FIFO, cap = max_thoughts)]
    M --> T
    T --> U[(Updated Thought Memory\n(B x L' x thought_dim))]

    %% Carry over to next step
    U -. feed next step .-> M
```

