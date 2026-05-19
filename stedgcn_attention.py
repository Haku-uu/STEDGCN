
import torch.nn as nn
import torch
# from torchinfo import summary


class TemporalScaledDotProductAttention(nn.Module):

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim#152
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)#[152,152]
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
       
        batch_size = query.shape[0]#16 #64
        tgt_length = query.shape[-2]#12 #170
        src_length = key.shape[-2]#12 #170

        query = self.FC_Q(query)#[64,6,170,152]
        key = self.FC_K(key)
        value = self.FC_V(value)


        key = key.transpose(
            -1, -2
        )  


        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        out = self.out_proj(out)#[64,6,170,152]

        return out

class TemporalMultiHeadAttention(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = TemporalScaledDotProductAttention(model_dim, num_heads, mask)
       
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        

        residual = out
        out = self.feed_forward(out)  
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out