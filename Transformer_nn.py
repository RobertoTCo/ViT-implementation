# Transformer encoder with MLP with GELU activation (Feedforward) and Multiheaded Self-Attention (MSA) blocks
# " The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self
# attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before
#  every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019)"
# https://arxiv.org/pdf/2010.11929

# Most of the code comes from the ViT Pytorch implementation https://github.com/lucidrains/vit-pytorch

from torch import nn
import torch

from einops import rearrange

# class FeedForward will be a MLP comprised of (a) LayerNrom, (b) Linear -> (c) GELU, (d) Linear,
# we add dropouts after every Linear layer
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
# class Multiheaded Self-Attention (MSA) will be comprised of parallel computation of Scaled Dot-Product Attention 
# for multiple-random initialization of Q, K, V for each token.
# It is done as (a) Normalisation, (b) Linear transformation to Q, K, V, (c) Split into n heads, (d) Scaled Dot-Product Attention, (e) concatenation & linear transformation

# It can be coded as one unique computation of qkv of dimension (dim_head * n heads) and then split into Q, K, V
# then, follow the Scaled Dot-Product Attention blocks (a) Q*K(), (b) softmax, (c) QK*V, (d) linear transformation to original dim
# where the dim of matrixes Q, K and V are (n tokens, dim_head), for multiheaded computation translate as (dim, embedded dim: dim_head * n heads)

class MSA(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.) -> None:
        super().__init__()
     
        embed_dim = dim_head *  heads # final (inner) dimension of the Q, K, V matrixes for all heads
        project_out = not (heads == 1 and dim_head == dim) # if we only have 1 head with dim == n tokens, the final linear layer is equal to Indentity

        self.heads = heads
        self.scale = dim_head ** -0.5 # scaling factor as the root of d

        self.norm = nn.LayerNorm(dim)

        self.soft_max_attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Linear(dim, embed_dim * 3, bias = False) # create a linear layer to project the input to Q, K, V

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (a) normalisation
        x = self.norm(x)
        # TODO: Move (d) as a proper nn.module for Scale dot product attention
        # (b) linear transformation to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # (c) split into n heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # we had the vector dimension d equal to dim_head * n heads, we split by n_head
        # (d) Scaled Dot-Product Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # Q*K
        attn = self.soft_max_attend(dots) # softmax + dropout (opt.)
        out = torch.matmul(attn, v) # QK*V
        # (e) concatenation & linear transformation
        out = rearrange(out, 'b h n d -> b n (h d)')  # we concatenate the heads again as the embedded dimension
        return self.to_out(out) # linear transformation to original dim (or Identity if there is no need)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # we create blocks of MSA+MLP based on depth parameter
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                MSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # both MSA and MLP implement a norm first thing to the input
        for attn, ffn in self.blocks:
            x = attn(x) + x # + x as residual connection
            x = ffn(x) + x
        return x # The ViT package implements a norm to the output, this has be moved to the MLP head in ViT