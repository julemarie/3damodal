import torch
import torch.nn as nn

def patchify(x, patch_size=16):
        """
            Patchify a batch of feature tensors.
        """
        K, C, H, W = x.shape
        
        x = x.reshape(K, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)    # [K, H', W', p_H, p_W, C]
        x = x.reshape(K, -1, *x.shape[3:])   # [K, H'*W', p_H, p_W, C]
        x = x.reshape(K, x.shape[1], -1) # [K, H'*W', p_H*p_W*C]

        return x


class Attention(nn.Module):
    """
        (Multi-head) self-attention layer.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim*3)

    def __call__(self, x):
        B, D, N = x.shape # B: #batches, D: embed_dim, N: 3
        proj_dim = N // self.num_heads
        
        qkv = self.qkv(x).reshape(B, D, 3, self.num_heads, proj_dim).permute(2, 0, 3, 1, 4) # [3, B, num_heads, D, proj_dim]
        q, k, v = qkv[0], qkv[1], qkv[2] # shape: [B, num_heads, D, proj_dim]

        att_scores = q @ k.transpose(2,3) # B x num_heads x D x D
        att_scores_sm = torch.tensor(nn.functional.softmax(att_scores, -1))
        weighted_vals = v[:,:,:,None,:] * att_scores_sm.transpose(-2,-1)[:,:,:,:,None] # B x num_heads x D x D x proj_dim
        sum = weighted_vals.sum(dim=2) # B x num_heads x D x proj_dim
        
        out  = sum.reshape(B, D, N)
        return out
    
class CrossAttention(nn.Module):
    """
        (Multi-head) cross attention layer.
    """
    def __init__(self, dim, num_heads) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)

        self.kv = nn.Linear(dim, dim*2)

    def forward(self, x, y):
        B, D, N = x.shape # B: #batches, D: embed_dim, N: 3
        By, Dy, Ny = y.shape # By == B and Ny == N
        proj_dim = N // self.num_heads
        proj_dim_y = Ny // self.num_heads
        # get queries from x2 (embedded, encoded and masked future frame)
        q = self.q(y).reshape(By, Dy, self.num_heads, proj_dim_y).permute(0, 2, 1, 3)
        # get keys and values from unmasked frame
        kv = self.kv(x).reshape(B, D, 2, self.num_heads, proj_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        print(q.shape)
        print()
        print("K", k.shape)

        # from here it's the same as the self-attention
        att_scores = q @ k.transpose(2,3) # B x num_heads x N x N
        att_scores_sm = torch.tensor(nn.functional.softmax(att_scores, -1))
        weighted_vals = v[:,:,:,None,:] * att_scores_sm.transpose(-2,-1)[:,:,:,:,None] # B x num_heads x N x N x proj_dim
        sum = weighted_vals.sum(dim=2) # B x num_heads x N x proj_dim

        out  = sum.reshape(By, Dy, Ny)
        return out


if __name__ == "__main__":
        x = torch.ones((2, 3, 32, 32))
        x = patchify(x, 16)

        
        print(x.shape)


