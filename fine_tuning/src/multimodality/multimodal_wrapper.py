
from torch import nn

"""
    Wrapper for a natively multimodal encoder
"""
class MultimodalWrapper(nn.Module):
    def __init__(self,
                 encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        B, n, H, W, D = x.shape

        z = self.encoder(x)   # (B*n, E) or (B*n, S, E) for ViTs with S nb of tokens
        
        # For MoE encoders, discard experts scores
        if isinstance(z, tuple):
            z = z[0]

        if z.ndim == 3:
            z = z.mean(dim=1) # (B*n, E)

        return z