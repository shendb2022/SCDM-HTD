import torch
from torch import nn
import math
from einops import rearrange, repeat
from torch.nn import init
'''
This part for designing the network to predict the adding noise in DDPM
'''

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(1, depth + 1):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        # self.initialize()

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

class ConditionalEmbedding(nn.Module):
    def __init__(self, band, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            # nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(band, d_model),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        # self.initialize()
        # self.condEmbedding = nn.Sequential(
        #     # nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
        #     nn.Linear(band, d_model),
        #     Rearrange('b c -> b 1 c'),
        #     nn.Conv1d(1, 64, kernel_size=50, stride=50),
        #     Rearrange('b c n -> b n c'),
        #     Mamba(64, 16, 1),
        #     Rearrange('b n c -> b (n c)'),
        #     nn.Linear(d_model//50*64, dim),
        #     Swish(),
        #     nn.Linear(dim, dim),
        # )

    def forward(self, t):
        emb = self.condEmbedding(t)
        # emb = self.net1(t)
        # print(emb.shape)
        return emb
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

class ResBlock(nn.Module):
    '''
    We change the 2D-CNN into 1D ones to extract the spectral features
    '''
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def __init__(self, in_ch, out_ch, tdim, dropout):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),  # Channels are divided into groups to calculate the LN
            Swish(),
            # nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1)
            nn.Linear(in_ch, out_ch)
        )
        # self.block1_linear = nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1)
        # self.out_ch = out_ch
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        # self.block2 = nn.Sequential(
        #     nn.GroupNorm(32, out_ch),
        #     Swish(),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1),
        # )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            # nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.Linear(out_ch, out_ch)
        )
        # self.block2_linear = nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1)
        # if in_ch != out_ch:
        #     ## If the input and output channels are not equal, a transform using convolution is required
        #     self.shortcut = nn.Conv1d(in_ch, out_ch, 1, stride=1, padding=0)
        # else:
        #     self.shortcut = nn.Identity()
        self.shortcut = nn.Identity()
        # self.initialize()
        ## We remove the self attention
        # if attn:
        #     self.attn = AttnBlock(out_ch)
        # else:
        #     self.attn = nn.Identity()
        # self.attn = nn.Sequential(
        #     Rearrange('b c n -> b n c'),
        #     Mamba(out_ch, 16, 2),
        #     # Transformer(out_ch, 1, 4, 64, 64),
        #     Rearrange('b c n -> b n c')
        # )
        # self.attn = Transformer(out_ch, 1, 4, 64, 64)
    def forward(self, x, temb, labels):
        h = self.block1(x)  ## The first block
        # conv1d_weights = self.block1_linear.weight  # (ch, 1, 3)
        # conv1d_bias = self.block1_linear.bias
        # kernel_weights = conv1d_weights[:, :, 1]
        # print(h.shape)
        # print(kernel_weights.shape)
        # print(conv1d_bias.shape)
        # h = torch.matmul(h, kernel_weights.T)+ conv1d_bias.view(1, self.out_ch)

        h += self.temb_proj(temb)[:, :]  ## Adding the time embedding projection
        h += self.cond_proj(labels)[:, :]  ## Adding the condition embedding projection
        h = self.block2(h)  ## The second block
        # conv1d_weights2 = self.block2_linear.weight  # (ch, 1, 3)
        # conv1d_bias2 = self.block2_linear.bias
        # kernel_weights2 = conv1d_weights2[:, :, 1]
        # # print(kernel_weights.shape)
        # h = torch.matmul(h, kernel_weights2.T) + conv1d_bias2.view(1, self.out_ch)

        h = h + self.shortcut(x)  ## The residual connection
        return h

class ResModule(nn.Module):
    def __init__(self, ch, tdim, dropout):
        super(ResModule, self).__init__()
        self.net1 = ResBlock(ch, ch, tdim, dropout)
        self.net2 = ResBlock(ch, ch, tdim, dropout)
        # self.avgpool = nn.AvgPool1d(2, 2)

    def forward(self, x, temb, labels):
        x = self.net1(x, temb, labels)
        x = self.net2(x, temb, labels)
        # x = self.avgpool(x)
        return x


class NetWork(nn.Module):
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        # init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
    def __init__(self, T, band, test_band, ch, dropout):
        super(NetWork, self).__init__()
        tdim = ch * 4
        ## The time embedding keeps the original form using sin and cos
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        ## The conditional embedding changes since concrete centers are provided
        self.cond_embedding = ConditionalEmbedding(test_band, ch, tdim)
        ## The head manipulating changes to the 1-D convolution
        # self.head = nn.Conv1d(1, ch, kernel_size=3, stride=1, padding=1, )
        self.head = nn.Linear(1, ch)

        self.ch = ch
        self.reslayers = nn.ModuleList()
        self.layers = 1
        # out_band = test_band
        for _ in range(1, self.layers + 1):
            self.reslayers.append(ResModule(ch, tdim, dropout))
            # out_band = out_band // 2
        # self.mean_pool = nn.AvgPool1d(out_band, out_band)
        self.out_adjust = nn.Sequential(
            nn.Linear(ch*band, band)
        )
        # self.initialize()

    def forward(self, x, t, center):
        # Timestep embedding
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(center)
        # Downsampling
        h = self.head(x)
        for ll in self.reslayers:
            h = ll(h, temb, cemb)
        # h = self.mean_pool(h)
        h = h.view(h.shape[0],-1)
        h = self.out_adjust(h)
        return h


if __name__ == '__main__':
    pass
