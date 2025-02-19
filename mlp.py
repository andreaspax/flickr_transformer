import torch


class MLP(torch.nn.Module):
    def __init__(self, d_model=64, dff=1024, dropout=0.1):
        super().__init__()

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(d_model, dff),
            torch.nn.GELU(),
            torch.nn.Linear(dff, d_model),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.linear(x)
