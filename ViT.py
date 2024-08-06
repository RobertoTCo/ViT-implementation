import torch.nn as nn
import torch
from einops.layers.torch import Rearrange # Fearrange: reorganize tensor dimensions (rearrange as a nn.module; 
from einops import repeat # repeat: repeat tensor values along a dimension

from Transformer_nn import TransformerEncoderBlock # Transformer Encoder Block for the ViT model

# Function for no-learning positional encoding using sine-cosine
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# This class can be used to check the shape of a previous layer output
class PrintSize(nn.Module):
  def __init__(self):
    super(PrintSize, self).__init__()
    
  def forward(self, x):
    print(x.shape)
    return x

# ViT implementation with Transformer and RoPE positional encoding (opt.)
class MyViT(nn.Module):
    def __init__(self, 
              num_classes: int, # number of classes in the dataset (final MLP head)
              img_size: tuple, # h, w format; , H: height, W: width
              num_channels: int, # C: num channels in input tensor
              patch_size: tuple, # h, w format; H: height, W: width
              depth: int, # Number of transformer blocks (MSA+MLP)
              dim: int, # Last dimension of output tensor after linear transformation, this is the final resolution for pixels in each patch (token dim d)
              dim_head: int, # dimension of the heads in the MSA (dim T) -> matrixes Q, K, V in Transformer are Txd dimensions
              heads: int, # number of heads in the MSA
              mlp_dim: int, # dimension of the hidden layer in the MLP
              dropout: float = 0., # dropout rate for layers in Transformer blocks
              embedding_dropout: float = 0., # dropout rate for the tokens+class token+positional encoding before the transformer
              ) -> None:
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "Img shape not entirely divisible by patch size"

        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        patch_dim = num_channels * patch_size[0] * patch_size[1]
        
        # sequence of tramsformations to linearly embed the patches
            # Rearrange creates a special nn module that organises tensor shape into a new one:
            # b c (h p1) (w p2) -> we´ll give a tensor of size: batch (n image), channels, img height = h * patch height, img width = w * patch width
            # *resulting h and w are the output of the division of the input image size by the patch size -> number of patches per row and col
            # b (h w) (p1 p2 c) -> we´ll convert the tensor to size: batch (n image), h*w, patch height * patch width * channels (patch dim)
            # This pattern creates a tensor that organises for every image a 2D matrix which each row is a patch and each column a pixel of the patch (vector)
            # Example: 
            # - 4 imgs of c, h, w: 3, 32, 32; patch size: 4, 4; 
            # - requires an tensor of size 4, 3, 32 = 8*4, 32= 8*4 = 4, 3, 32, 32 -> prod: 12288
            # - output: this tensor is rearranged into 4, 8*8, 4*4*3 = 4, 64, 48 -> prod: 12288
            # Example rectangular 1 img 1 channel 32x16, patch size 4x4,
            # - requires a tensor of size 1, 1, 32 = 8*4, 16 = 4*4; -> prod: 512
            # - output: this tensor is rearranged into 1, 8*4, 4*4*1 = 1, 32, 16 -> prod: 512
            # Example 1 img 1 b&w channel 0 to 255: 1, 1, 28= 7*4, 28= 7*4; patch size: 4, 4 -> output: 1, 49, 16 # MNIST case
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        ) # the final dimension is set my dim parameter which can reduce the embedding token size (d)

        self.class_token = nn.Parameter(torch.randn(1, 1, dim)) # learnable parameter for each pixel of a patch

        # The ViT pytorch package makes the positional encoding learnable by the backpropagation
        #self.positional_encoding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # learnable parameter, num_patches + 1 to add an extra token
        # I´ll prefer to try the sine-cosine positional encoding using RoPE
        # https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
        # This will be implemented as a nn module using Rotary Positional Embeddings (RoPE) 
        self.positional_encoding = posemb_sincos_2d(h = img_size[0] // patch_size[0] , w = img_size[1] // patch_size[1], dim = dim)

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.transformer = TransformerEncoderBlock(dim = dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = dropout)
        
        # self.to_latent = nn.Identity() ¿?
        self.mlp_head = nn.Sequential(
           nn.LayerNorm(dim),
           nn.Linear(dim, num_classes)
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(image)
        # TODO: add the positional encoding, class token, transformer, mlp head, and output layer

        # due to dim matching, we can add positional encoding to the tokens before concatenating the class_token
        device = image.device
        tokens += self.positional_encoding.to(device, dtype=tokens.dtype) # uncomment to use RoPE positional encoding , 
      
        # class token is added to the patches in the last dim
          # 1. repeat the class token to match the length of the bacth of images in the first dim
          # 2. add the class token to the patches in the row dim (1)
        class_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b = tokens.shape[0])

        tokens = torch.cat((class_tokens, tokens), dim = 1)
        # positional encoding is added to the patches+cls token
        #tokens += self.positional_encoding[:, :(tokens.shape[1] + 1)] # uncomment to use learnable positional encoding
              
        tokens = self.embedding_dropout(tokens)
        x = self.transformer(tokens)[:, 0] # TODO: evaluate mean pooling method

        # x = self.to_latent(x) # ¿?
        return self.mlp_head(x)

if __name__ == '__main__':
  # Current model
  rows = 32
  cols = 16
  model = MyViT(
      num_classes=3,
      img_size=(rows, cols),
      num_channels = 1,
      patch_size=(4, 4),
      dim=16, # final resolution for pixels in each patch (embedded vectors (tokens) dim d)
      depth = 2, # Number of transformer blocks (MSA+MLP)
      dim_head = 16, # dimension of the heads in the MSA (dim T) -> matrixes Q, K, V in Transformer are Txd dimensions
      heads = 2, # number of heads in the MSA
      mlp_dim = 2, # dimension of the hidden layer in the MLP
      dropout = 0.1, # dropout rate for layers in Transformer blocks
      embedding_dropout = 0.1 # dropout rate for the tokens+class token+positional encoding before the transformer
  )
  x = torch.randn(4, 1, rows, cols) # Dummy images
  print(model(x).shape)
  print(model)
  