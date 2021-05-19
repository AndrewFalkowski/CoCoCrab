import numpy as np
import pandas as pd

import torch
from torch import nn


# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


# %%
class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super(ResidualNetwork, self).__init__()
        dims = [input_dim]+hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'


class Embedder(nn.Module):
    def __init__(self,
                 d_model,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'data/element_properties'
        # # Choose what element information the model receives
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding
        # mat2vec = f'{elem_dir}/onehot.csv'  # onehot encoding (atomic number)
        # mat2vec = f'{elem_dir}/random_200.csv'  # random vec for elements

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


# %%
class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 d_model,
                 resolution=100,
                 log10=False,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        # x = torch.linspace(0, self.resolution - 1,
        #                    self.resolution,
        #                    requires_grad=False) \
        #     .view(self.resolution, 1)
        # fraction = torch.linspace(0, self.d_model - 1,
        #                           self.d_model,
        #                           requires_grad=False) \
        #     .view(1, self.d_model).repeat(self.resolution, 1)
        #
        # pe = torch.zeros(self.resolution, self.d_model)
        # pe[:, 0::2] = torch.sin(x /torch.pow(
        #     50,2 * fraction[:, 0::2] / self.d_model))
        # pe[:, 1::2] = torch.cos(x / torch.pow(
        #     50, 2 * fraction[:, 1::2] / self.d_model))
        # pe = self.register_buffer('pe', pe)

    def forward(self, x):

        rounded_frac = x.clone().to(self.compute_device)
        # rounded_frac = x
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x[x > 1] = 1
            # x = 1 - x  # for sinusoidal encoding at x=0
            
        x[x < 1/self.resolution] = 1/self.resolution
        
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=True).view(1, self.d_model) \
                                  .to(self.compute_device)

        pe = torch.zeros(rounded_frac.shape[0],
                          rounded_frac.shape[1],
                          self.d_model).to(self.compute_device)

        for i in range(pe.shape[0]):

            pe[i, :, 0::2] = (torch.sin(rounded_frac[i].view(1,-1) /torch.pow(
                50,2 * fraction[:, 0::2].T / self.d_model))).T

            pe[i, :, 1::2] = (torch.cos(rounded_frac[i].view(1,-1) / torch.pow(
                50, 2 * fraction[:, 1::2].T / self.d_model))).T
            
        return pe


# %%
class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 frac=False,
                 attn=True,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device
        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False, compute_device=self.compute_device)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True, compute_device=self.compute_device)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(self.d_model,
                                                       nhead=self.heads,
                                                       dim_feedforward=2048,
                                                       dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                             num_layers=self.N)

    def forward(self, src, frac):
        x = self.embed(src) * 2**self.emb_scaler
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1
        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2**(1-self.pos_scaler)**2
        ple_scaler = 2**(1-self.pos_scaler_log)**2
        pe[:, :, :self.d_model//2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model//2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src,
                                         src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)

        return x


# %%
class CrabNet(nn.Module):
    def __init__(self,
                 out_dims=3,
                 d_model=512,
                 N=3,
                 heads=4,
                 compute_device=None):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.encoder = Encoder(d_model=self.d_model,
                               N=self.N,
                               heads=self.heads,
                               compute_device=self.compute_device)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model,
                                         self.out_dims,
                                         self.out_hidden)

    def forward(self, src, frac):
        output = self.encoder(src, frac)

        # average the "element contribution" at the end
        # mask so you only average "elements"
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1)/(~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output


# def frac_expansion(frac, d_model=200, resolution=100, log10=False):

#     d_model = d_model//2

#     # preprocess fractions
#     if log10:
#         frac = 0.0025* (torch.log2(frac))**2
#         frac[frac > 1] = 1
#     frac[frac < 1/resolution] = 1/resolution
#     rounded_frac = torch.round(frac * resolution).to(dtype=torch.long) - 1

#     fraction = torch.linspace(0, d_model - 1,
#                               d_model,
#                               requires_grad=False) \
#                               .view(1, d_model)

#     pe = torch.zeros(rounded_frac.shape[0], d_model)

#     pe[:, 0::2] = torch.sin(rounded_frac /torch.pow(
#         50,2 * fraction[:, 0::2] / d_model))

#     pe[:, 1::2] = torch.cos(rounded_frac / torch.pow(
#         50, 2 * fraction[:, 1::2] / d_model))

#     pe = pe.view(-1,1,100)
#     return pe

# %%
if __name__ == '__main__':
    model = CrabNet()
