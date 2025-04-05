import sys
sys.path.append("/home/wdh/seld/IEEEStinfor/src/training/asteroid/models")


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
# from src.training.asteroid.models.conv_tasnet import ConvTasNet

# class Net(nn.Module):
#     def __init__(self, n_src = 2):
#         super(Net, self).__init__()
#         self.n_src = n_src
#         self.sepnet = ConvTasNet(n_src = self.n_src, num_mics = 1)
#         self.encoder = nn.Sequential(
#                     nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1),
#                     nn.LeakyReLU())
#     def forward(self, mix):
#         # mix = mix[:, :1]
#         mix = self.encoder(mix)
#         output = self.sepnet(mix) #[B, 2, M, 48000]
#         if len(output.shape) < 4:
#             output = output.unsqueeze(2)
#         return output


import torch
from torch.autograd import Variable
from src.models import conv_tasnet_models_attention_set as models_attention_set


class Net(torch.nn.Module):
    def __init__(self, model_mic_num: int = 4,
                model_ch_dim: int = 8,
                model_enc_dim: int = 512,
                model_feature_dim: int = 128,
                model_win: int = 16,
                model_layer: int = 8,
                model_stack: int = 1,
                model_kernel: int = 3,
                model_num_spk: int = 2,
                model_causal: bool = False):
        super(Net, self).__init__()
        # hyper parameters
        self.mic_num = model_mic_num
        self.num_spk = model_num_spk

        # increased enc dim
        self.enc_dim = model_enc_dim
        self.feature_dim = model_feature_dim

        self.ch_dim = model_ch_dim

        self.win = int(16000 * model_win / 1000)
        self.stride = self.win // 2

        self.layer = model_layer
        self.stack = model_stack
        self.kernel = model_kernel

        self.causal = model_causal

        # input encoder
        self.encoder = torch.nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)

        # TCN separator
        self.TCN = models_attention_set.TCN(self.mic_num, self.ch_dim, self.enc_dim, self.enc_dim * self.num_spk,
                                            self.feature_dim, self.feature_dim * 4,  # single modified
                                            self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = torch.nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, input):
        # Padding
        output, rest = self.pad_signal(input)

        batch_size = output.size(0)
        num_ch = output.size(1)
        enc_output = self.encoder(output.view(batch_size * num_ch, 1, -1)).view(batch_size, num_ch, self.enc_dim, -1)  # B, C, N, L

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L

        # reference mic = mic5
        masked_output = enc_output[:, 0:0+1] * masks

        # waveform decoder
        output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L

        output = output.view(batch_size, self.num_spk, -1)  # B, C, T

        return output.unsqueeze(2)

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

