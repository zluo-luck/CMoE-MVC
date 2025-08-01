import torch
import torch.nn.functional as F


class HypersphereLoss(torch.nn.Module):
    """
    Implementation of the loss described in 'Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere.' [0]
    [0] Tongzhou Wang. et.al, 2020, ... https://arxiv.org/abs/2005.10242
    Adapt from https://github.com/lightly-ai/lightly/blob/9bda4ee1b8bd756da9e4430197b26077861893ed/lightly/loss/hypersphere_loss.py
    """

    def __init__(self, t=1.0, lam=1.0, alpha=2.0):
        super(HypersphereLoss, self).__init__()
        self.t = t
        self.lam = lam
        self.alpha = alpha

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        x = F.normalize(z_a)
        y = F.normalize(z_b)

        def lalign(x, y):
            return (x - y).norm(dim=1).pow(self.alpha).mean()

        # def lunif(x):
        #     sq_pdist = torch.pdist(x, p=2).pow(2)
        #     return sq_pdist.mul(-self.t).exp().mean().log()

        return lalign(x, y)