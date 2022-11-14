import torch
import torch.nn as nn

def extract_span_features_with_minus(hidden_state):
    """
    Borrowed from https://github.com/nikitakit/self-attentive-parser/blob/master/src/benepar/parse_chart.py
    Args:
        hidden_state: bs * (N + 2) * H

    Returns:
        (N + 1) * (N + 1) * bs * hidden_size
    """
    _, _, d_model = hidden_state.size()
    fencepost_annotations = torch.cat(
        [
            hidden_state[:, :-1, : d_model // 2],
            hidden_state[:, 1:, d_model // 2 :],
        ],
        -1,
    )
    span_features = torch.unsqueeze(fencepost_annotations, 1) - torch.unsqueeze(
        fencepost_annotations, 2
    )
    return span_features.permute(1, 2, 0, 3)

class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin2(torch.nn.functional.relu(self.lin1(x)))) + x

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.s_dim = hidden_size
        self.mlp = nn.Sequential(nn.Linear(input_size * 2, self.s_dim), ResLayer(self.s_dim, self.s_dim), ResLayer(self.s_dim, self.s_dim), nn.Linear(self.s_dim, output_size))

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], -1)
        return self.mlp(x)

def gen_truncated_geometric(n, p, remove_first=False):
    """
    Generate a truncated geometric distribution with n samples and p probability of success.

    Example:
        n = 3, remove_first False,  [p, p(1-p), (1-p)^2]
        n = 3, remove_first True,  [0, p, (1-p)]

        n = k, remove_first False,  [p, p(1-p), ..., (1-p)^(k-1)]
        n = k, remove_first True,  [0, p, (1-p), ..., (1-p)^(k-2)]
    """
    # corner case
    if n == 1:
        return [0.0] if remove_first else [1.0]

    res = []
    for i in range(1, n + 1):
        if i != n:
            if remove_first:
                if i == 1:
                    res.append(0)
                else:
                    res.append(p * (1 - p) ** (i - 2))
            else:
                res.append(p * (1 - p) ** (i - 1))
        else:
            if remove_first:
                res.append((1 - p) ** (i - 2))
            else:
                res.append((1 - p) ** (i - 1))

    return res
