import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4, causal=False):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.scale = torch.sqrt(torch.tensor(self.d_k)).item()

        self.keys = torch.nn.Linear(d_model, 3 * d_model)

        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_model, d_model)
        self.causal = causal

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # Split into three equal chunks of size d_model each
        qry, key, val = self.keys(x).chunk(3, dim=-1)

        # dim: batch_size, seq_len, d_model -> batch_size, seq_len, heads, d_k
        qry = qry.reshape(batch_size, seq_len, self.heads, self.d_k)
        key = key.reshape(batch_size, seq_len, self.heads, self.d_k)
        val = val.reshape(batch_size, seq_len, self.heads, self.d_k)

        # dim: batch_size, seq_len, heads, d_k -> batch_size, heads, seq_len, d_k
        qry = qry.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)

        A = torch.matmul(qry, key.transpose(-2, -1)) / self.scale


        if self.causal:
                mask = torch.tril(torch.ones(A.shape[-2:], device=A.device))
                mask = mask.unsqueeze(0)  # add batch dimension
                A = A.masked_fill(mask == 0, float("-inf"))
            
        A = torch.softmax(A, dim=-1)
        A = torch.matmul(A, val)  # dim: batch_size, heads, seq_len, d_k

        A = A.transpose(1, 2)  # dim: batch_size, seq_len, heads, d_k
        A = A.reshape(batch_size, seq_len, self.d_model)

        A = self.out(A)
        A = self.dropout(A)
        A = A + x  # residual connection
        A = self.layer_norm(A)

        return A