import torch
from contextlib import nullcontext
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32) # (knots,)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device) # (D, num_proj)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        # proj @ A = (T, B, D) @ (D, num_proj) = (T, B, num_proj)
        x_t = (proj @ A).unsqueeze(-1) * self.t # (T, B, num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square() # (T, num_proj, knots)
        statistic = (err @ self.weights) * proj.size(-2) # (T, num_proj)
        return statistic.mean() # average over projections and time


def _normalize_orthogonal_mode(mode):
    """Map config aliases (row, space) to internal projector modes."""
    mode = str(mode).lower()
    aliases = {
        "row": "row-ortho",
        "row-ortho": "row-ortho",
        "space": "space-ortho",
        "space-ortho": "space-ortho",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unknown orthogonal mode {mode!r}; use 'row' or 'space'."
        )
    return aliases[mode]


def _build_subspace_projectors(
    embed_dim, subspace_dim, num_subspaces, mode="row-ortho", seed=None
):
    """
    Return W: (num_subspaces, subspace_dim, embed_dim) SubJEPA-style projectors.

    row-ortho: each W_i has orthonormal rows; subspaces may overlap.
    space-ortho: disjoint blocks from one global orthogonal basis.
    """
    mode = _normalize_orthogonal_mode(mode)
    rng_ctx = torch.random.fork_rng() if seed is not None else nullcontext()
    with rng_ctx:
        if seed is not None:
            torch.manual_seed(seed)
        if mode == "row-ortho":
            Ws = []
            for _ in range(num_subspaces):
                q, _ = torch.linalg.qr(
                    torch.randn(embed_dim, subspace_dim), mode="reduced"
                )
                Ws.append(q.T)  # (subspace_dim, embed_dim)
            return torch.stack(Ws, dim=0)

        total_dim = num_subspaces * subspace_dim
        if total_dim > embed_dim:
            raise ValueError(
                f"space-ortho subspaces need num_subspaces * subspace_dim <= embed_dim, "
                f"got {num_subspaces} * {subspace_dim} = {total_dim} > {embed_dim}"
            )
        q, _ = torch.linalg.qr(
            torch.randn(embed_dim, total_dim), mode="reduced"
        )
        Ws = []
        for k in range(num_subspaces):
            sl = slice(k * subspace_dim, (k + 1) * subspace_dim)
            Ws.append(q[:, sl].T)
        return torch.stack(Ws, dim=0)


class SubspaceSIGReg(nn.Module):
    """SIGReg on K frozen random orthonormal subspace projections."""

    def __init__(self, embed_dim, subspace_dim, num_subspaces, sigreg, mode="row-ortho"):
        super().__init__()
        self.embed_dim = embed_dim
        self.subspace_dim = subspace_dim
        self.num_subspaces = num_subspaces
        self.mode = mode.lower()
        self.sigreg = sigreg

        W = _build_subspace_projectors(
            embed_dim, subspace_dim, num_subspaces, mode=self.mode
        )
        self.register_buffer("W", W)  # (K, subspace_dim, embed_dim)

    def forward(self, emb):
        """
        emb: (B, T, embed_dim)
        """
        losses = []
        for k in range(self.num_subspaces):
            # W[k]: (subspace_dim, embed_dim) — project emb into subspace k
            sub_emb = emb @ self.W[k].T  # (B, T, subspace_dim)
            loss_k = self.sigreg(sub_emb.transpose(0, 1))  # SIGReg expects (T, B, subspace_dim)
            losses.append(loss_k)
        return torch.stack(losses).mean()


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True, attn_mask=None):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=False
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=causal
            )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None, attn_mask=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            if isinstance(block, Block):
                x = block(x, attn_mask=attn_mask)
            else:
                x = block(x, c, attn_mask=attn_mask)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


def _visible_keys_from_mask(mask, query_idx):
    """Return key indices allowed for a query row."""
    neg_inf = torch.finfo(mask.dtype).min if mask.dtype.is_floating_point else float("-inf")
    return (mask[query_idx] > neg_inf / 2).nonzero(as_tuple=False).squeeze(-1)


def _assert_block_causal_mask_sanity():
    """Validate block-causal mask: visibility depends on frame index."""
    T, K = 3, 4
    mask = TokenARPredictor.make_block_causal_mask(
        T, K, device=torch.device("cpu"), dtype=torch.float32
    )

    q_t1_k0 = 1 * K + 0
    q_t1_k3 = 1 * K + 3
    q_t2_k1 = 2 * K + 1

    vis_t1_k0 = _visible_keys_from_mask(mask, q_t1_k0)
    vis_t1_k3 = _visible_keys_from_mask(mask, q_t1_k3)
    vis_t2_k1 = _visible_keys_from_mask(mask, q_t2_k1)

    expected_t1 = torch.arange((1 + 1) * K, dtype=torch.long)  # frames 0 and 1
    expected_t2 = torch.arange((2 + 1) * K, dtype=torch.long)  # frames 0, 1, 2

    assert torch.equal(vis_t1_k0, expected_t1), vis_t1_k0
    assert torch.equal(vis_t1_k3, expected_t1), vis_t1_k3
    assert torch.equal(vis_t1_k0, vis_t1_k3)
    assert torch.equal(vis_t2_k1, expected_t2), vis_t2_k1

    frame2_keys = torch.arange(2 * K, T * K, dtype=torch.long)
    assert not any(k in vis_t1_k0.tolist() for k in frame2_keys.tolist())


class TokenARPredictor(nn.Module):
    """Autoregressive predictor over coordinate-aligned subspace tokens.

    ARPredictor uses T tokens of dimension D. TokenARPredictor splits each
    D-dimensional token into K coordinate-aligned subspace tokens of dimension d,
    and runs the same ConditionalBlock Transformer over T*K tokens.
    The block-causal mask is used to ensure that the transformer only attends to
    tokens in the same or earlier frames.
    Action conditioning is preserved via AdaLN-zero by projecting action embeddings
    into token-aligned condition tokens.
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        action_dim=None,
        subspace_dim=32,
        num_subspaces=None,
        residual=True,
    ):
        super().__init__()
        action_dim = input_dim if action_dim is None else action_dim
        if num_subspaces is None:
            if input_dim % subspace_dim != 0:
                raise ValueError(
                    f"num_subspaces=None requires input_dim % subspace_dim == 0, "
                    f"got {input_dim} % {subspace_dim}"
                )
            num_subspaces = input_dim // subspace_dim
        if num_subspaces * subspace_dim != input_dim:
            raise ValueError(
                f"require num_subspaces * subspace_dim == input_dim, "
                f"got {num_subspaces} * {subspace_dim} != {input_dim}"
            )

        self.input_dim = input_dim
        self.output_dim = input_dim
        self.action_dim = action_dim
        self.subspace_dim = subspace_dim
        self.num_subspaces = num_subspaces
        self.num_frames = num_frames
        self.residual = residual

        self.time_embedding = nn.Parameter(
            torch.zeros(1, num_frames, 1, subspace_dim)
        )
        self.subspace_embedding = nn.Parameter(
            torch.zeros(1, 1, num_subspaces, subspace_dim)
        )
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.subspace_embedding, std=0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.cond_proj = nn.Linear(action_dim, num_subspaces * subspace_dim)
        self.transformer = Transformer(
            input_dim=subspace_dim,
            hidden_dim=hidden_dim,
            output_dim=subspace_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=ConditionalBlock,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(subspace_dim),
            nn.Linear(subspace_dim, subspace_dim),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)
        # _assert_block_causal_mask_sanity()

    @staticmethod
    def make_block_causal_mask(T, K, device, dtype):
        idx = torch.arange(T * K, device=device)
        q_frame = idx[:, None] // K
        k_frame = idx[None, :] // K
        allowed = k_frame <= q_frame

        mask = torch.zeros(T * K, T * K, device=device, dtype=dtype)
        neg_inf = torch.finfo(dtype).min if dtype.is_floating_point else float("-inf")
        mask = mask.masked_fill(~allowed, neg_inf)
        return mask

    def forward(self, x, c):
        """
        x: [B, T, D]
        c: [B, T, action_dim]
        return: [B, T, D]
        """
        B, T, D = x.shape
        K = self.num_subspaces
        d = self.subspace_dim
        if D != self.input_dim:
            raise ValueError(f"Expected x last dim {self.input_dim}, got {D}")
        if T > self.num_frames:
            raise ValueError(f"Expected T <= num_frames ({self.num_frames}), got T={T}")
        if c.shape[0] != B or c.shape[1] != T or c.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected c shape [B, T, {self.action_dim}] matching x batch/time, "
                f"got {tuple(c.shape)}"
            )

        x_tok = x.contiguous().view(B, T, K, d)
        cond_tok = self.cond_proj(c).view(B, T, K, d)

        h = x_tok + self.time_embedding[:, :T] + self.subspace_embedding
        h = self.dropout(h)
        h = h.reshape(B, T * K, d)
        cond_tok = cond_tok.reshape(B, T * K, d)
        attn_mask = self.make_block_causal_mask(T, K, h.device, h.dtype)
        h = self.transformer(h, cond_tok, attn_mask=attn_mask)
        h = h.view(B, T, K, d)

        delta = self.out_proj(h)
        y_tok = x_tok + delta if self.residual else delta
        return y_tok.flatten(-2)
