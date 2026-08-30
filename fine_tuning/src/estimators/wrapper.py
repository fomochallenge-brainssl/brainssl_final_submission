from nidl.estimators import BaseEstimator, TransformerMixin
from torch import nn
import torch
from typing import Optional, Any

class EstimatorWrapper(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        encoder: nn.Module,
        is_fitted: bool,
        **kwargs: Any,
    ):
        ignore = kwargs.pop("ignore", ["callbacks"])
        if "callbacks" not in ignore:
            ignore.append("callbacks")
        if isinstance(encoder, nn.Module) and "encoder" not in ignore:
            ignore.append("encoder")

        super().__init__(**kwargs, ignore=ignore)
        
        if is_fitted:
            self.fitted_ = True
        else:
            raise ValueError('Only already fitted encoders should be used with this wraper. If the encoder is' \
            'indeed already fitted please set `is_fitted` to `True`. Otherwise use another estimator from' \
            '`nidl.estimators` to fit your encoder.')

        self.encoder = encoder

    def transform_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Encode the input data into the latent space.

        Parameters
        ----------
        batch: torch.Tensor
            A batch of data. This is given as is to the encoder.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        features: torch.Tensor
            The encoded features returned by the encoder.

        """
        return self.encoder(batch)