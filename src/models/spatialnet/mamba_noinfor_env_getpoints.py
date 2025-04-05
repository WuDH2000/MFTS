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
from src.training.env_utils import get_env_points
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
        self.stft = STFT(n_fft=256, n_hop=128, win_len=256)
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, 
                                 kernel_size=encoder_kernel_size, stride=1, padding="same")

        # spatialnet layers
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
        # print('input_x', x.shape)
        mix_env = get_env_points(x[:, 0:1]) #[B, 1, T]
        X, stft_paras = self.stft.stft(x)  # [B,C,F,T], complex
        B, C, F, T = X.shape
        X = X.permute(0, 2, 3, 1)  # B,F,T,C; complex
        X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2C
        B, F, T, H0 = X.shape
        # print(x.reshape(B * F, T, H0).permute(0, 2, 1).shape)
        x = self.encoder(X.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        H = x.shape[2]

        x = x.reshape(B, F, T, H)
        for m in self.layers:
            x, attn = m(x)
        out = self.decoder(x) #[B, F, T, 2*ns]
        # print(out.shape)
        if not torch.is_complex(out):
            out = torch.view_as_complex(out.float().reshape(B, F, T, -1, 2))  # [B,F,T,Spk]
        out = out.permute(0, 3, 1, 2)  # [B,Spk,F,T]
        out = self.stft.istft(out, stft_paras) #[B, 2, 48000]
        out_points = get_env_points(out) #[B, 2, 48000]
        return [out_points, mix_env]

# Define optimizer, loss and metrics

def optimizer(model, data_parallel=False, **kwargs):
    return optim.Adam(model.parameters(), **kwargs)

def nmse_loss(pred, tgt, mix_env):
    # getloss = 10 * torch.log10((torch.mean((pred -tgt)** 2,dim =(-1))+ 1e-10)
    #                         /(torch.mean(tgt **2,dim=(-1))+ 1e-10)+ 1e-10)#[batch]
    # B = getloss.shape[0]# batch sizefor b in range(B):tgt b= tgt[b]
    # for bb in range(B):
    #     tgt_b = tgt[bb]
    #     pred_b = pred[bb]
    #     if torch.sum(tgt_b ** 2) < 1e-5:
    #         label_b = mix_env[bb]
    #         getloss[bb] =  10 * torch.log10(torch.mean((pred_b)** 2)+ 
    #                                     0.01 * torch.mean(label_b ** 2)+ 1e-10)
    getloss = (torch.mean((pred -tgt)** 2,dim =(-1))+ 1e-10) #[B]
    tgtnorm = (torch.mean(tgt ** 2, dim = -1) + 1e-10) #[B]
    zero_label = tgtnorm < 1e-5 #[B] True: nosrc
    tgtnorm[zero_label] = 1
    getloss = getloss / tgtnorm
    # print(getloss.shape)
    
    mixnorm = 0.01 * torch.mean(mix_env ** 2, axis = -1).squeeze(-1)
    # print(mixnorm.shape)
    mixnorm[~zero_label] = 0
    getloss = getloss + mixnorm
    getloss = 10 * torch.log10(getloss + 1e-10)
    return getloss

pitmix = PITLossWrapper(nmse_loss, pit_from="pw_pt")
def loss(pred, tgt, return_est = False):
    pred_env = pred[0]
    mix_env = pred[1]
    
    if return_est:
        get_loss, permuted_pred = pitmix(pred_env, tgt, mix_env = mix_env, return_est = True)
        return get_loss, permuted_pred
    else:
        get_loss = pitmix(pred_env, tgt, mix_env = mix_env)
        return get_loss

def nmse_metrics(pred, tgt):
    #pred:list
    #tgt:[B, ns, T]
    # print(len(pred))
    _, pred = loss(pred, tgt, return_est=True)
    # print(pred.shape)
    # print(tgt.shape)
    B = pred.shape[0]# batch sizefor b in range(B):tgt b= tgt[b]
    rec_nmse = []
    rec_acc = 0
    for bb in range(B):
        tgt_b = tgt[bb] #[ns, T]
        pred_b = pred[bb] #[ns, T]

        pred_exist = torch.max(pred_b, axis = -1)[0] > 0.25  #[max_ns]
        tgt_exist = torch.max(tgt_b, axis = -1)[0] > 0.25  #[max_ns]
        if torch.sum((pred_exist == tgt_exist).int()) == tgt_b.shape[0]:
            rec_acc += 1
        
        exist_tensor = pred_exist & tgt_exist
        tgt_b_s = tgt_b[exist_tensor]
        pred_b_s = pred_b[exist_tensor]
        if tgt_b_s.shape[0] == 0:
            continue
        get_nmse_s = 10 * torch.log10((torch.mean((pred_b_s - tgt_b_s) ** 2, axis = -1))
                            / (torch.mean(tgt_b_s ** 2, axis = -1)) + 1e-10).mean()
        rec_nmse.append(get_nmse_s.item())
    rec_nmse = torch.mean(torch.tensor(rec_nmse))
    rec_acc = rec_acc / B
    return rec_nmse, rec_acc

def metrics(output, gt):
    """ Function to compute metrics """
    metrics = {}
    rec_nmse, rec_acc = nmse_metrics(output, gt)
    metrics['nmse'] = [rec_nmse.cpu().detach().numpy()]
    metrics['acc'] = [rec_acc]
    return metrics

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    x = torch.randn(8, 4, 48000).cuda()
    model = Net().cuda()
    z = model(x)
    print(z[0].shape)
    
    tgt = torch.randn(size = (8, 2, 376)).cuda()
    getloss = loss(z, tgt)
    print(getloss)
    getmetrics = metrics(z, tgt)
    print(getmetrics)