
from torch import nn

"""
    Wrapper to adapt a unimodal encoder (expected input shape: (B, 1, H, W, D))
    to a multimodal task (actual input shape: (B, n, H, W, D) n > 1).

    Modalities (i.e. channels) are encoded separately and then averaged 
    to compute a single latent representation for each subject (output shape: (B, E))
"""
class UnimodalWrapper(nn.Module):
    def __init__(self,
                 encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        B, n, H, W, D = x.shape

        # encode each modality independently
        x = x.reshape(B * n, 1, H, W, D)

        z = self.encoder(x)   # (B*n, E) or (B*n, S, E) for ViTs with S nb of tokens

        # For MoE encoders, discard experts scores
        if isinstance(z, tuple):
            z = z[0]

        if z.ndim == 3:
            z = z.mean(dim=1)

        # restore modality dimension
        z = z.reshape(B, n, -1)          # (B, n, E)

        # late fusion
        z = z.mean(dim=1)            # (B, E)
        return z