import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class MultiHeadAttention(nn.Module):
    def __init__(self, inputq_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0 # embed_dim has to be divisible by num_heads
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = embed_dim // num_heads

        self.q = nn.Linear(inputq_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, 2*embed_dim)

        self.out_proj = nn.Linear(embed_dim, inputq_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """ Initialization according to original Transformer. """
        nn.init.xavier_uniform_(self.q.weight)
        self.q.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv.weight)
        self.kv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v):
        d_k = q.shape[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, input_q, input_kv):
        B, N, D = input_kv.shape # B: batch_size, N: seq_length, D: embed_dim
        _, Nq, Dq = input_q.shape

        # Get queries from input_q (learnable mask queries of the different masks)
        q = self.q(input_q).reshape(B, Nq, self.num_heads, self.proj_dim).permute(0, 2, 1, 3)

        # Get keys and valuees from input_kv (the roi_embedding)
        kv = self.kv(input_kv)
        kv = kv.reshape(B, N, self.num_heads, 2*self.proj_dim)
        kv = kv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        k, v = kv.chunk(2, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(B, Nq, self.embed_dim)
        out = self.out_proj(values)

        return out

if __name__ == "__main__":
    x = torch.ones((5,3,512,512))
    # x = x.flatten(2).permute(0,2,1) # [K, H*W, C]
    K, N, D  = x.shape
    Q = torch.randn((1,D,3))
    Q = Q.repeat((K, 1, 1))



    attn = MultiHeadAttention(3, 3, 1)

    vals = attn(Q, Q)




