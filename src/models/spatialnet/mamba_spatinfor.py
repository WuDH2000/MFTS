import math
from collections import OrderedDict
from typing import Optional
from typing import *
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
from asteroid.losses import PITLossWrapper
from asteroid.losses import pairwise_neg_sisdr
from mamba_ssm import Mamba
from .spat_utils.norm import *
from .spat_utils.linear_group import *
from src.training.stft import STFT

class SpatialNetLayer(nn.Module):
    def __init__(
            self,
            dim_hidden: int,
            dim_squeeze: int,
            num_freqs: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full: nn.Module = None,
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # narrow-band block
        # MHSA module
        self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        # self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.mamba = nn.Sequential(
            nn.Linear(dim_hidden, 32),
            Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=32, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            ),
            nn.Linear(32, dim_hidden)
        )
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module
        
        self.norm_tconv = new_norm(norms[1], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        self.mamba_tconv = nn.Sequential(
            nn.Linear(dim_hidden, 32),
            Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=32, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            ),
            nn.SiLU(),
            nn.Linear(32, dim_hidden)
        )
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        x_, attn = self._tsa(x, att_mask)
        x = x + x_
        x = x + self._tconvffn(x)
        return x, attn

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        _B, _F, _T, _H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(_B * _F, _T, _H)
        x = self.mamba(x)
        x = x.reshape(_B, _F, _T, _H)
        return self.dropout_mhsa(x), None

    def _tconvffn(self, x: Tensor) -> Tensor:
        _B, _F, _T, _H0 = x.shape
        x = self.norm_tconv(x)
        x = x.reshape(_B * _F, _T, _H0)
        x = self.mamba_tconv(x) #[B*F, T, H0]
        x = x.reshape(_B, _F, _T, _H0)
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        _B, _F, _T, _H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(_B * _T, _H, _F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(_B, _T, _H, _F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        _B, _F, _T, _H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(_B * _T, _H, _F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(_B, _T, -1, _F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(_B * _T, -1, _F)

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(_B, _T, _H, _F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"

class Net(nn.Module):
    def __init__(
            self,
            dim_input = 8,
            dim_output = 2 * 4, 
            num_layers = 8,# 12 for large
            encoder_kernel_size = 5,
            dim_hidden = 96, # 192 for large
            dropout = [0, 0, 0],
            kernel_size = [5, 3],
            conv_groups = [8, 8],
            norms = ["LN", "LN", "GN", "LN", "LN", "LN"],
            dim_squeeze = 8, # 16 for large
            num_freqs = 129,
            full_share = 0,
            padding = 'zeros'
    ):
        super(Net, self).__init__()
        self.num_freq = num_freqs
        self.stft = STFT(n_fft=256, n_hop=128, win_len=256)
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, 
                                 kernel_size=encoder_kernel_size, stride=1, padding="same")
        self.spat_encoder = nn.Conv1d(in_channels=3, out_channels=dim_hidden, 
                                 kernel_size=encoder_kernel_size, stride=1, padding="same")
        full = None
        layers = []
        for l in range(num_layers):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

    def forward(self, x):
        #x:[B, 2M, F, T]
        #spatinf:[B, 3, T]
        #wav_inf:[B, 2M, F, T]
        #yuandaima:x: [Batch, Freq, Time, Feature]
        x, spat_inf = x[0], x[1]
        X, stft_paras = self.stft.stft(x)  # [B,C,F,T], complex
        B, C, F, T = X.shape
        X = X.permute(0, 2, 3, 1)  # B,F,T,C; complex
        X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2C
        B, F, T, H0 = X.shape
        # print(x.reshape(B * F, T, H0).permute(0, 2, 1).shape)
        x = self.encoder(X.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        H = x.shape[2]

        x = x.reshape(B, F, T, H)
        
        cond_inf = None
        if spat_inf is not None:
            cond_inf = self.spat_encoder(spat_inf).permute(0, 2, 1) #[B, T, enc_dim]
            cond_inf = cond_inf.unsqueeze(1) #[B, 1, T, enc_dim]
            x = x * cond_inf
        
        if spat_inf is not None:
            for m in self.layers:
                x, attn = m(x)
        out = self.decoder(x) #[B, F, T, 2M]
        
        if not torch.is_complex(out):
            out = torch.view_as_complex(out.float().reshape(B, F, T, -1, 2))  # [B,F,T,Spk]
            
        out = out.permute(0, 3, 1, 2)  # [B,Spk,F,T]
        out = self.stft.istft(out, stft_paras) #[B, 4, 48000]
        return out.contiguous()


# Define optimizer, loss and metrics

def optimizer(model, data_parallel=False, **kwargs):
    return optim.Adam(model.parameters(), **kwargs)
# def loss(pred, tgt):
#     # return -si_snr(pred, tgt).mean()
#     return (-si_snr(pred, tgt).mean()) * 0.1 + (-snr(pred, tgt).mean()) * 0.9
def loss(pred, tgt):
    # return -si_snr(pred, tgt).mean()
    return -snr(pred, tgt).mean()

def metrics(output, gt):
    """ Function to compute metrics """
    metrics = {}
    
    metrics['sisnr'] = [si_snr(output, gt).mean().cpu().detach().numpy()]
    metrics['snr'] = [snr(output, gt).mean().cpu().detach().numpy()]
    return metrics

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    x = torch.randn(8, 4, 48000).cuda()
    spat_inf = torch.randn(8, 3, 376).cuda()
    # print(x.shape)
    model = Net().cuda()
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters()) / (1024 * 1024)}')
    z = model([x, spat_inf])
    print(z.shape)
   
    label = torch.rand(8, 4, 48000).cuda()
    getloss = loss(z, label)
    print(getloss)
    
    getmetrics = metrics(z, label)
    print(getmetrics)
