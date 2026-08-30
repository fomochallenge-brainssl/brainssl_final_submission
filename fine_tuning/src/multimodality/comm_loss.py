import torch.nn.functional as func
import torch
import torch.nn as nn
from src.utils.comm_utils import all_gather_batch_with_grad, all_gather_batch


class CoMMLoss(nn.Module):
    """
        Copied from https://github.com/Duplums/CoMM/tree/mainhttps://github.com/Duplums/CoMM/tree/main
        
        Normalized Temperature Cross-Entropy Loss for Multi-Modal Contrastive Learning as defined in CoMM [1]

        [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self, temperature=0.1, weights=None):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        self.INF = 1e8

    def infonce(self, z1, z2):
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z 
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        # Prepare embeddings (normalize + gather across all GPU)
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        assert len(z1) == len(z2)
        n_emb = len(z1)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:]

        # Apply InfoNCE between a "prototype embedding" and all the others
        loss = []
        acc = []
        for i in range(n_emb):
            loss1, acc1 = self.infonce(z1[i], z2[prototype])
            loss2, acc2 = self.infonce(z2[i], z1[prototype])
            loss.append((loss1 + loss2) / 2.)
            acc.append((acc1 + acc2) / 2.)
        ssl_acc = {"ssl_acc_%i"%i: acc_ for i, acc_ in enumerate(acc)}
        losses = {"ssl_loss_%i"%i: l for i, l in enumerate(loss)}
        if self.weights is not None:
            loss = torch.mean(torch.stack(loss) * torch.tensor(self.weights, device=z1[0].device))
        else:
            loss = torch.mean(torch.stack(loss))
        acc = torch.mean(torch.stack(acc))
        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class MaskedCoMMLoss(CoMMLoss):
    """CoMM loss for missing modality case.

    Restrict multimodal InfoNCE to subjects who have modality i and the prototype
    indexed to the same subjects, so the positive pairs stay aligned. The prototype alignment is
    defined for every subject and uses the whole batch.

    """
    def __init__(self, temperature=0.1, weights=None, gather=False):
        super().__init__(temperature=temperature, weights=weights)
        self.gather = gather

    def forward(self, outputs):
        """
        :param outputs: Dict with keys:
                - "aug1_embed"
                - "aug2_embed"
                - "prototype"
                - "present", bool tensor (bsize, n_modalities), True where the
                  subject has that modality.
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        present = outputs["present"]
        assert len(z1) == len(z2)
        n_emb = len(z1)
        prototype = prototype % n_emb           # -1 -> last (the multimodal one)
        n_mod = n_emb - 1
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1] #list of size M+1 with each fused embedding of shape (B,D)
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]

        if self.gather:
            Z = all_gather_batch_with_grad(z1 + z2)          # keeps grad to local rank
            z1, z2 = Z[:n_emb], Z[n_emb:]
            present = all_gather_batch([present.float()])[0].bool()   # mask: no grad

        # Apply InfoNCE between the "prototype embedding" and all the others,
        # over the subjects for which each subset is defined.
        loss = []
        acc = []
        kept = []
        for i in range(n_emb):
            if i < n_mod:
                valid = present[:, i]           # subjects having modality i
            else:
                valid = torch.ones(z1[i].shape[0], dtype=torch.bool,
                                   device=z1[i].device)     # prototype: all
            if int(valid.sum()) < 2:            # InfoNCE needs >= 2 samples
                continue
            loss1, acc1 = self.infonce(z1[i][valid], z2[prototype][valid])
            loss2, acc2 = self.infonce(z2[i][valid], z1[prototype][valid])
            loss.append((loss1 + loss2) / 2.)
            acc.append((acc1 + acc2) / 2.)
            kept.append(i)

        ssl_acc = {"ssl_acc_%i"%i: acc_ for i, acc_ in zip(kept, acc)}
        losses = {"ssl_loss_%i"%i: l for i, l in zip(kept, loss)}
        if self.weights is not None:
            weights = torch.tensor([self.weights[i] for i in kept],
                                   device=z1[0].device)
            loss = torch.mean(torch.stack(loss) * weights)
        else:
            loss = torch.mean(torch.stack(loss))
        acc = torch.mean(torch.stack(acc))
        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}