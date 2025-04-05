from src.training.stinfor_dataset import SpkUnkEnvPITDataset as infer_dataset
import src.models.spatialnet.mamba_noinfor_env_getpoints as network_rs
import argparse
import torch
from src.helpers import utils
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from math import factorial
import scipy.special as sp
import scipy.signal as ss
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional import signal_noise_ratio as snr
from torchmetrics.functional import signal_distortion_ratio as sdr
from src.training.traj_utils.sph import sph2cart, cart2sph, angular_error, mvdr_sh_freq
from src.training.env_utils import get_env_points, get_env, get_env_by_inpterp
import sys
sys.path.append("./src/models/audio_source_separation-main/src")
from bss.mnmf import FastMultichannelISNMF as MNMF

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(os.getpid())

app = '_nmf_env'
print(app)

metric_dict = dict()
# metric_dict['num_srcs'] = []
# metric_dict['est_srcs'] = []
metric_dict['nmse'] = []
# metric_dict['acc'] = []
# metric_dict['recall'] = []
# metric_dict['precision'] = []

parser = argparse.ArgumentParser()
exp_dir = "experiments/invest"
args = parser.parse_args()
params = utils.Params(os.path.join(exp_dir, 'config.json'))
for k, v in params.__dict__.items():
    vars(args)[k] = v
use_cuda = torch.cuda.is_available()
kwargs = {
        'num_workers': 0,
        'pin_memory': True
    }
data_test = infer_dataset(**args.test_data)
test_loader = torch.utils.data.DataLoader(data_test,
                                               batch_size=1,
                                               shuffle=False, **kwargs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fft_size, hop_size = 4096, 2048
def model_rs(mixture, nsrc = 2):
    x = mixture[0].cpu().detach().numpy()
    n_channels, T = x.shape
    n_sources = 2
    _, _, X = ss.stft(x, nperseg=fft_size, noverlap=fft_size-hop_size)
    mnmf = MNMF(n_basis=2, n_sources=nsrc)
    Y = mnmf(X, iteration=100)
    _, y = ss.istft(Y, nperseg=fft_size, noverlap=fft_size-hop_size)
    # print(y.shape)
    y = torch.from_numpy(y[:,:T]).to(mixture.device)
    y = y.unsqueeze(0)
    return get_env_points(y)

def nmse(pred, tgt):
    #[B, s, 48000]
    #[s, 48000]
    return 10 * torch.log10(torch.sum((tgt - pred) ** 2, axis = -1) / torch.sum(tgt ** 2, axis = -1))

with torch.no_grad():
    for batch_idx, (mixed, tgt_env_p) in \
                    enumerate(tqdm(test_loader, desc='Test', ncols=100)):
        
        rec_nmse = []
        
        mixed = mixed.to(device) #[B, 4, 48000]
        tgt_env_p = tgt_env_p.to(device) #[B, maxns, T]
        
        est_env_p = model_rs(mixed) #[B, maxns, T]
        # print(est_env_p.shape)
        # print(tgt_env_p.shape)
        _, est_env_p = network_rs.loss([est_env_p, get_env_points(mixed[:, 0:1])], tgt_env_p, return_est=True)

        get_nmse_s = 10 * torch.log10((torch.mean((est_env_p - tgt_env_p) ** 2, axis = -1))
                            / (torch.mean(tgt_env_p ** 2, axis = -1)) + 1e-10).mean()
        print('nmse', get_nmse_s.item())
        metric_dict['nmse'].append(get_nmse_s.item())


np.save('./res_dict/metric_dict_iter'+app+'.npy', metric_dict)
print('./res_dict/metric_dict_iter'+app+'.npy')

metric_dict = np.load('./res_dict/metric_dict_iter'+app+'.npy', allow_pickle=True)
metric_dict = metric_dict.item()

get_nmse = np.array(metric_dict['nmse']).reshape(-1, 1)
print(get_nmse.shape)
print(np.mean(get_nmse))

# def get_xy(val_x, val_y, prec = 100):
#     y_x = dict()
#     for batch_idx in range(val_x.shape[0]):
#         for iter_idx in range(val_x.shape[1]):
#             x_idx = int(val_x[batch_idx, iter_idx] * prec)
#             if not x_idx in y_x:
#                 y_x[x_idx] = []
#             y_x[x_idx].append(val_y[batch_idx, iter_idx])

#     xs, ys = [], []
#     # print(res_dict['pow'])
#     for xx in y_x:
#         xs.append(xx / prec)
#         ys.append(np.mean(np.array(y_x[xx])))
#     # print(t60s)
#     idx = [i[0] for i in sorted(enumerate(xs), key=lambda x:x[1])]#[5:]
#     xs = np.array(xs)[idx]
#     ys = np.array(ys)[idx]
#     return xs, ys, y_x

# x,y, _ = get_xy(get_tgtsrcs, get_nmse)
# print(x, y)
# x,y, _ = get_xy(get_tgtsrcs, get_acc)
# print(x, y)
# x,y, _ = get_xy(get_tgtsrcs, get_recall)
# print(x, y)
# x,y, _ = get_xy(get_tgtsrcs, get_prec)
# print(x, y)

# def get_stotas(val_x, val_y):
#     y_x = dict()
#     for batch_idx in range(val_x.shape[0]):
#         for iter_idx in range(val_x.shape[1]):
#             x_idx = int(val_x[batch_idx, iter_idx])
#             if not x_idx in y_x:
#                 y_x[x_idx] = dict()
#             y_idx = val_y[batch_idx, iter_idx]
#             if not y_idx in y_x[x_idx]:
#                 y_x[x_idx][y_idx] = 0
#             y_x[x_idx][y_idx] += 1
#     return y_x

# y_x = get_stotas(get_tgtsrcs, get_estsrcs)
# print(y_x)

