import torch
import mlp
import attention


class Decoder(torch.nn.Module):
    def __init__(self, di_initial=512, d_model=512, dff=2048, vocab_size=49408, seq_len=77, dropout=0.1):
        super().__init__()

        self.positional_encoding = torch.nn.parameter.Parameter(torch.randn(seq_len, d_model))
        self.ln = torch.nn.LayerNorm(d_model)
        
        self.decoder_blocks = torch.nn.ModuleList([
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
        ])

        self.final = torch.nn.Linear(d_model, vocab_size)

        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        
        y = y.unsqueeze(1)  # Change shape from (batch_size, d_model) to (batch_size, 1, d_model)

        x = torch.cat((y, x), dim=1)  # Concatenate y at the start of x along the sequence length dimension

        seq_len = x.size(1)
        # Only use positional encoding up to current sequence length

        x = x + self.positional_encoding[:seq_len, :]

        x = self.ln(x)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.final(x)
        return x
    
class DecoderBlock(torch.nn.Module):
    def __init__(self, dff=1024, d_model=64, dropout=0.1):
        super().__init__()

        
        self.attention = attention.SelfAttention(
            d_model, 
            dropout=dropout, 
            heads=16, 
            causal=True
        )
        self.mlp = mlp.MLP(d_model, dff, dropout)


    def forward(self, x: torch.Tensor):

        x = self.attention(x)

        x = x + self.mlp(x)
        return x
    
    