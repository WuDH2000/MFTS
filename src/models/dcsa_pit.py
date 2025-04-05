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
from .dcc_tf_stinfor import mod_pad
from asteroid.losses import PITLossWrapper

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride,
                      padding, groups=in_channels, dilation=dilation),
            # LayerNormPermuted(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1,
                      padding=0),
            # LayerNormPermuted(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class DilatedConvEncoder(nn.Module):
    """
    A dilated causal convolution based encoder for encoding
    time domain audio input into latent space.
    """
    def __init__(self, channels, num_layers, kernel_size=3):
        super(DilatedConvEncoder, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Kernel size must be odd."

        # Dilated causal conv layers aggregate previous context to obtain
        # contexful encoded input.
        _dcc_layers = OrderedDict()
        for i in range(num_layers):
            dcc_layer = DepthwiseSeparableConv(
                channels, channels, kernel_size=3, stride=1,
                padding=(kernel_size // 2) * 2**i, dilation=2**i)
            _dcc_layers.update({'dcc_%d' % i: dcc_layer})
        self.dcc_layers = nn.Sequential(_dcc_layers)

    def forward(self, x):
        for layer in self.dcc_layers:
            # print(layer(x).shape)
            x = x + layer(x)
        return x

class LinearTransformerDecoder(nn.Module):
    """
    A casual transformer decoder which decodes input vectors using
    precisely `ctx_len` past vectors in the sequence, and using no future
    vectors at all.
    """
    def __init__(self, model_dim, chunk_size, num_layers,
                 nhead, use_pos_enc, ff_dim):
        super(LinearTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.chunk_size = chunk_size
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.unfold = nn.Unfold(kernel_size=(3 * chunk_size, 1), stride=chunk_size)
        self.pos_enc = PositionalEncoding(model_dim, max_len=10 * chunk_size)
        self.tf_dec_layers = nn.ModuleList([nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim,
            batch_first=True) for _ in range(num_layers)])

    def _permute_and_unfold(self, x):
        """
        Unfolds the sequence into a batch of sequences.

        Args:
            x: [B, C, L]
        Returns:
            [B * (L // chunk_size), 3 * chunk_size, C]
        """
        B, C, L = x.shape
        # print(x.shape)  #[2, 128, 6100]
        x = F.pad(x, (self.chunk_size, self.chunk_size)) # [B, C, L + 2 * chunk_size]
        # print(x.shape)  #[2, 128, 6300]
        x = self.unfold(x.unsqueeze(-1)) # [B, C * 3 * chunk_size, -1]
        # print(x.shape)  #[2, 38400, 61]
        x = x.view(B, C, 3 * self.chunk_size, -1).permute(0, 3, 2, 1) # [B, -1, 3 * chunk_size, C]
        x = x.reshape(-1, 3 * self.chunk_size, C) # [B * (L // chunk_size), 3 * chunk_size, C]
        return x

    def forward(self, mem, K=1000):
        """
        Args:
            mem: [B, model_dim, T] from x
            K: Number of chunks to process at a time to avoid OOM.
        """
        # print(mem.shape) #[2, 128, 6001]
        mem, mod = mod_pad(mem, self.chunk_size, (0, 0))
        # print(mem.shape)  #[2, 128, 6100]
        # Input sequence length
        B, C, T = mem.shape

        mem = self._permute_and_unfold(mem) # [B * (T // chunk_size), 3 * chunk_size, C]
        # print(tgt.shape)  #在T维度划分成3*chunksize的块，attention在这个块上做

        # Positional encoding
        if self.use_pos_enc:
            mem = mem + self.pos_enc(mem)
            
        for i, tf_dec_layer in enumerate(self.tf_dec_layers):
            _mem = torch.zeros_like(mem)
            for i in range(int(math.ceil(mem.shape[0] / K))):
                _mem[i*K:(i+1)*K] = tf_dec_layer(mem[i*K:(i+1)*K], mem[i*K:(i+1)*K])
            mem = _mem
            
    # Permute back to [B, C, T]
        mem = mem[:, self.chunk_size:-self.chunk_size, :]
        mem = mem.reshape(B, -1, C) # [B, T, C]
        mem = mem.permute(0, 2, 1)  #[B, C, T]
        # print(tgt.shape)
        if mod != 0:
            mem = mem[..., :-mod]
            # print(tgt.shape)
        return mem

class MaskNet(nn.Module):
    def __init__(self, enc_dim, num_enc_layers, dec_dim, dec_chunk_size,
                 num_dec_layers, use_pos_enc, skip_connection, proj):
        super(MaskNet, self).__init__()
        self.skip_connection = skip_connection
        self.proj = proj

        # Encoder based on dilated causal convolutions.
        self.encoder = DilatedConvEncoder(channels=enc_dim,
                                          num_layers=num_enc_layers)

        # Project between encoder and decoder dimensions
        self.proj_e2d_e = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())
        self.proj_d2e = nn.Sequential(
            nn.Conv1d(dec_dim, enc_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())

        # Transformer decoder that operates on chunks of size
        # buffer size.
        self.decoder = LinearTransformerDecoder(
            model_dim=dec_dim, chunk_size=dec_chunk_size, num_layers=num_dec_layers,
            nhead=8, use_pos_enc=use_pos_enc, ff_dim=2 * dec_dim)

    def forward(self, x):
        # print(mode)
        """
        Generates a mask based on encoded input `e` and the one-hot
        label `label`.

        Args:
            x: [B, C, T]
                Input audio sequence
            l: [B, C, T]
                Label embedding
        """
        # Enocder the label integrated input
        # print(x.shape)  #[2, 256, 6001]
        e = self.encoder(x)
        # print(e.shape)  #[2, 512, 6001]
        if self.proj:
            e = self.proj_e2d_e(e)
            # print(m.shape)
            # Cross-attention to predict the mask
            # print(mode)
            m = self.decoder(e)
        else:
            # Cross-attention to predict the mask
            m = self.decoder(e)

        # Project mask to encoder dimensions
        if self.proj:
            m = self.proj_d2e(m)
        return m

class Net(nn.Module):
    def __init__(self, L=32,
                 enc_dim=512, num_enc_layers=10,
                 dec_dim=256, num_dec_layers=1,
                 dec_chunk_size=13, num_mics = 4, use_pos_enc=True, skip_connection=True,
                 proj=True, lookahead=True, loc_onsep = True, n_src = 2):
        super(Net, self).__init__()
        self.L = L
        self.enc_dim = enc_dim
        self.lookahead = lookahead
        self.num_mics = num_mics
        self.n_src = n_src

        # Input conv to convert input audio to a latent representation
        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(
                nn.Conv1d(in_channels=num_mics,
                        out_channels=enc_dim, kernel_size=kernel_size, stride=L,
                        padding=0, bias = False),
                nn.ReLU())
        # Mask generator
        self.mask_gen = MaskNet(
            enc_dim=enc_dim, num_enc_layers=num_enc_layers,
            dec_dim=dec_dim, dec_chunk_size=dec_chunk_size,
            num_dec_layers=num_dec_layers, use_pos_enc=use_pos_enc,
            skip_connection=skip_connection, proj=proj)

        # Output conv layer
        self.out_conv_t = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=enc_dim, out_channels=self.n_src * 3,
                kernel_size=3 * L,
                stride=L,
                padding=L, bias=False),
            nn.Tanh())

    def forward(self, x):
        """
        Extracts the audio corresponding to the `label` in the given
        `mixture`. Generates `chunk_size` samples per iteration.

        Args:
            mixed: [B, n_mics, T]
                input audio mixture
        Returns:
            out: [B, n_spk, T]
                extracted audio with sounds corresponding to the `label`
        """
        # print(x.shape)
        # print(label.shape)
        mod = 0
        pad_size = (self.L, self.L) if self.lookahead else (0, 0)
        x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)
        # print(x.shape) #[2, 256, 6001]
        x = self.in_conv(x)
        m_x = self.mask_gen(x)
        # Apply mask and decode
        # x = x * m
        #x: [B, 128, T]
        x = self.out_conv_t(m_x)
        B, tt = x.shape[0], x.shape[-1]
        x = x.reshape(B, self.n_src, 3, tt) #[B,2,3,T]
        # print(x.shape)  #[B, 1, 3, 48008]
        # Remove mod padding, if present.
        if mod != 0:
            x = x[..., :-mod]
        return x.permute(0, 1, 3, 2) #[B, ns, T, 3]

# Define optimizer, loss and metrics

def optimizer(model, data_parallel=False, **kwargs):
    return optim.Adam(model.parameters(), **kwargs)

def mse_loss(pred, tgt):
    # pred[batch, t, 3]
    return torch.mean((pred - tgt) ** 2, dim = (-1, -2)) #[batch]

def loss_loc(pred, tgt, est_env = None):
    #[B, T, 3]
    loss_diff = 0
    step = 1
    while step < 2 ** 5:
        tloss = mse_loss(pred[:, step::step] - pred[:, 0:-step:step], 
                       tgt[:, step::step] - tgt[:, 0:-step:step])
        loss_diff += tloss
        step *= 2
    # getloss = 0.5 * loss_doa + 0.5 * loss_diff
    loss_doa = 0
    if est_env is not None:
        est_env = est_env.permute(0, 2, 1)
        # print(est_env.shape)
        # print(pred.shape)
        loss_doa = mse_loss(pred * est_env, tgt * est_env)
    else:
        loss_doa = mse_loss(pred, tgt)
    getloss = loss_doa * 0.5 + loss_diff * 0.5
    # get_loss = torch.nn.functional.mse_loss(pred, tgt)
    return getloss

pitmse = PITLossWrapper(loss_loc, pit_from="pw_pt")
def loss(pred, tgt, return_est = False):
    # print(pred.shape)
    # print(tgt.shape)
    return pitmse(pred, tgt, return_est = return_est)

def cart2sph(cart):
	xy2 = cart[..., 0]**2 + cart[..., 1]**2
	sph = torch.zeros_like(cart)
	sph[...,0] = 0#torch.sqrt(xy2 + cart[:, :,2]**2)
	sph[...,1] = torch.arctan2(torch.sqrt(xy2), cart[...,2]) # Elevation angle defined from Z-axis down
	sph[...,2] = torch.arctan2(cart[...,1], cart[...,0])
	return sph

def angular_error(the_pred, phi_pred, the_true, phi_true):
    """ Angular distance between spherical coordinates.
    """
    aux = torch.cos(the_true) * torch.cos(the_pred) + \
            torch.sin(the_true) * torch.sin(the_pred) * torch.cos(phi_true - phi_pred)
    return torch.acos(torch.clamp(aux, -0.99999, 0.99999))

def angle_diff(pred, tgt, keep_batch = False):
    # print(pred.shape)  #[B, ns, 48000, 3]
    # print(tgt.shape)  #[B, ns, 48000, 3]
    # get_norm: [B, ns, 1, 48000]
    pred = cart2sph(pred)  #[16, ns, tt, 3] ->[0, ele, azi]
    tgt = cart2sph(tgt)
    res_a = torch.sqrt(torch.mean(torch.pow(angular_error(pred[..., 1], 
            pred[..., 2], tgt[..., 1], tgt[..., 2]), 2), axis = -1)) / torch.pi * 180 #[B, ns]
    if keep_batch:
        return torch.mean(res_a, axis = -1)
    else:
        return res_a.mean()

def metrics(output, gt):
    """ Function to compute metrics """
    metrics = {}
    # print(output.shape) #[8, 2, 48000, 3]
    # print(gt.shape) #[8, 2, 48000, 3]
    _, output_perm = loss(output, gt, return_est = True)
    ad = angle_diff(output_perm, gt)
    metrics['ad'] = [ad.cpu().detach().numpy()]
    return metrics

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    x = torch.randn(4, 4, 48000).cuda()
    model = Net().cuda()
    z = model(x)
    print(z.shape)
    
    tgt = torch.randn(4, 2, 48000, 3).cuda()
    getloss = loss(z, tgt)
    getmetrics = metrics(z, tgt)
    print(getloss)
    print(getmetrics)
    
