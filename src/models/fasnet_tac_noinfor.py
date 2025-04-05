import sys
sys.path.append("/home/wdh/seld/IEEEStinfor/src/training")


import math
from collections import OrderedDict
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
from asteroid.losses import PITLossWrapper
from asteroid.losses import pairwise_neg_sisdr
from src.training.espnet2.enh.separator.fasnet_separator import FaSNetSeparator

class Net(nn.Module):
    def __init__(self,
                input_dim = 4,
                enc_dim = 64,
                feature_dim = 64,
                hidden_dim = 128,
                layer = 6,
                segment_size = 50,
                num_spk = 2,
                win_len = 16,
                context_len = 16,
                fasnet_type = 'fasnet'):
        super(Net, self).__init__()
        self.separator = FaSNetSeparator(input_dim,
                enc_dim,
                feature_dim,
                hidden_dim,
                layer,
                segment_size,
                num_spk,
                win_len,
                context_len,
                fasnet_type)
    def forward(self, mix):
        # print(mix.shape)
        ilens = torch.zeros(mix.shape[0]).to(mix.device)
        ilens[:] = mix.shape[-1]
        mix = mix.permute(0, 2, 1) #[B, T, M]
        output_list, _, _ = self.separator(mix, ilens) #[(B, T), ...]
        output =  torch.cat([i.unsqueeze(1).unsqueeze(1) for i in output_list], axis = 1)
        return output

def optimizer(model, data_parallel=False, **kwargs):
    return optim.Adam(model.parameters(), **kwargs)

# def mix_loss(pred, tgt):
#     # pred[batch, M, 48000]
#     # print('loss sisnr', torch.mean(si_snr(pred, tgt), dim = (-1)))
#     # print('loss snr', torch.mean(snr(pred, tgt), dim = (-1)))
#     return (0.1 * torch.mean(-si_snr(pred, tgt), dim = (-1)) +
#             0.9 * torch.mean(-snr(pred, tgt), dim = (-1)))#[batch]

def mix_loss(pred, tgt):
    # pred[batch, M, 48000]
    # print('loss sisnr', torch.mean(si_snr(pred, tgt), dim = (-1)))
    # print('loss snr', torch.mean(snr(pred, tgt), dim = (-1)))
    return torch.mean(-snr(pred, tgt), dim = (-1))#[batch]

pitmix = PITLossWrapper(mix_loss, pit_from="pw_pt")
def loss(pred, tgt, return_est = False):
    # print(pred.shape)
    # print(tgt.shape)
    return pitmix(pred, tgt, return_est = return_est)

def sisnr_loss(pred, tgt):
    # pred[batch, M, 48000]
    # print('metrics sisnr', torch.mean(si_snr(pred, tgt), dim = (-1)))
    return (torch.mean(-si_snr(pred, tgt), dim = (-1)))#[batch]

pitsisnr = PITLossWrapper(sisnr_loss, pit_from="pw_pt")
def sisnr_metrics(pred, tgt):
    # print(pred.shape)
    # print(tgt.shape)
    return -pitsisnr(pred, tgt)

def snr_loss(pred, tgt):
    # pred[batch, M, 48000]
    # print('metrics snr', torch.mean(snr(pred, tgt), dim = (-1)))
    return (torch.mean(-snr(pred, tgt), dim = (-1)))#[batch]

pitsnr = PITLossWrapper(snr_loss, pit_from="pw_pt")
def snr_metrics(pred, tgt):
    # print(pred.shape)
    # print(tgt.shape)
    return -pitsnr(pred, tgt)


def metrics(output, gt):
    """ Function to compute metrics """
    metrics = {}
    # print(output.shape) #[8, 2, M, 48000]
    # print(gt.shape) #[8, 2, M, 48000]
    # print(mixed.shape) #[8, 2, M, 48000]
    metrics['sisnr'] = [sisnr_metrics(output, gt).cpu().detach().numpy()]
    metrics['snr'] = [snr_metrics(output, gt).cpu().detach().numpy()]
    return metrics

# pitlossfunc = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
# def loss(pred, tgt):
#     return pitlossfunc(pred, tgt)

# def metrics(mixed, output, gt):
#     """ Function to compute metrics """
#     # print(mixed.shape)
#     # print(output.shape)
#     # print(gt.shape)
#     mixed = mixed[:, 0:1, :]
#     mixed = torch.cat([mixed, mixed], axis = 1)
#     metrics = {}
#     metrics['scale_invariant_signal_noise_ratio'] = [((-loss(output, gt)) - (-loss(mixed, gt))).cpu().detach().numpy()]
#     return metrics

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    x = torch.randn(4, 4, 48000).cuda()
    # print(x.shape)
    model = Net().cuda()
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters()) / (1024 * 1024)}')
    z = model(x)
    print(z.shape)
    
    label = torch.rand(4, 2, 1, 48000).cuda()
    getloss = loss(z, label)
    print(getloss)
    
    getmetrics = metrics(z, label)
    print(getmetrics)
    
    # model = nn.MultiheadAttention(embed_dim=32, num_heads=4, kdim=16, vdim=32, batch_first=True).cuda()
    # q = torch.randn(8, 100, 16).cuda()
    # k = torch.randn(8, 100, 16).cuda()
    # v = torch.randn(8, 100, 32).cuda()
    # z, _ = model(q, k, v)
    # print(z.shape)

