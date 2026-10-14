"""Stage II emotional text adaptation module (paper Sec. 3.2 / 4.1).

Fully-connected layers of sizes 772, 600, 500, 400, 300, 200, 100, 50, 40 and 16
with ReLU, mapping a 768-d emotion text embedding to logits over the GST tokens.
Trained as soft-label multi-class classification against the GST weights.
"""
import torch
import torch.nn.functional as F
from torch import nn


class AdaptationModule(nn.Module):
    def __init__(self, in_dim=768, hidden=(772, 600, 500, 400, 300, 200, 100, 50, 40), n_tokens=16):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_tokens))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits

    def predict_weights(self, x):
        return self.forward(x).softmax(-1)


def soft_cross_entropy(logits, target):
    """Categorical cross-entropy with soft labels: -sum_k t_k log softmax(z)_k."""
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
