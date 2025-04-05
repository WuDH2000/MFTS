from src.training.stinfor_dataset import MFTSDataset as infer_dataset
# import src.models.spatialnet.mamba_noinfor as network_rs
# import src.models.conv_tasnet_noinfor as network_rs
import src.models.fasnet_tac_noinfor as network_rs
import src.models.dcsa as network_wl
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(os.getpid())

app = '_fasnettac_sep_frame'
print(app)

metric_dict = dict()

metric_dict['nmse'] = []
metric_dict['sch_sisnr'] = []
metric_dict['sch_snr'] = []
metric_dict['sch_sdr'] = []
metric_dict['t60'] = []
metric_dict['spat_sep'] = []
metric_dict['snr_frame'] = []

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
                                               batch_size=16,
                                               shuffle=False, **kwargs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_rs = network_rs.Net().to(device)
# model_rs_path = 'experiments/trans_noinfor/spatialnet_nsrc2_ssch/48.pt'
# model_rs_path = 'experiments/trans_noinfor/conv_tasnet_nsrc2_ssch/59.pt'
model_rs_path = 'experiments/trans_noinfor/fasnettac_nsrc2_ssch/58.pt'
model_rs.load_state_dict(
    torch.load(model_rs_path, map_location=device)["model_state_dict"]
)

with torch.no_grad():
    for batch_idx, (mixed, clean_wav, traj_t, _, T60) in \
                    enumerate(tqdm(test_loader, desc='Test', ncols=100)):
        rec_schsnr = []
        rec_schsdr = []
        rec_schsisnr = []
        
        mixed = mixed.to(device) #[B, 4, 48000]
        clean_wav = clean_wav.to(device) #[B, maxns, 4, len]
        
        est_wav = model_rs(mixed)#[B, maxns, 1, T]
        # print(est_env_p.shape)
        # print(tgt_env_p.shape)
        _, est_sch_wav = network_rs.loss(est_wav, clean_wav[:, :, 0:1], return_est=True)
        # est_sch_wav = est_sch_wav[:, 0]
        # clean_wav = clean_wav[:, 0]

        get_snr = snr(est_sch_wav[:, :, 0], clean_wav[:, :, 0]).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
        rec_schsnr.append(get_snr)
        print('schsnr', get_snr)
        get_sdr = sdr(est_sch_wav[:, :, 0], clean_wav[:, :, 0]).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
        rec_schsdr.append(get_sdr)
        print('schsdr', get_sdr)
        get_sisnr = si_snr(est_sch_wav[:, :, 0], clean_wav[:, :, 0]).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
        rec_schsisnr.append(get_sisnr)
        print('schsisnr', get_sisnr)
        metric_dict['t60'].append(T60.cpu().detach().numpy().reshape(-1, 1))
        metric_dict['sch_sisnr'].append(np.concatenate(rec_schsisnr, axis = -1))
        metric_dict['sch_snr'].append(np.concatenate(rec_schsnr, axis = -1))
        metric_dict['sch_sdr'].append(np.concatenate(rec_schsdr, axis = -1))
        
        traj_sph = network_wl.cart2sph(traj_t) #[B, ns, len, 3]
        spat_sep = network_wl.angular_error(
                    traj_sph[:, 0, :, 1], 
                    traj_sph[:, 0, :, 2],
                    traj_sph[:, 1, :, 1],
                    traj_sph[:, 1, :, 2]) / torch.pi * 180 #[B, len]
        
        B = est_sch_wav.shape[0]
        est_sch_wav_frame = est_sch_wav[:, :, 0].reshape(B, 2, 125, -1)
        clean_wav_frame = clean_wav[:, :, 0].reshape(B, 2, 125, -1)
        rec_spatsep = torch.mean(spat_sep.reshape(B, 125, -1), axis = -1) #[B, 125]
        snr_frame = torch.mean(snr(est_sch_wav_frame, clean_wav_frame), axis = 1) #[B, 125]
        metric_dict['spat_sep'].append(rec_spatsep.cpu().detach().numpy())
        metric_dict['snr_frame'].append(snr_frame.cpu().detach().numpy())
        
        # if (batch_idx + 1) % 2 == 0:
        #     break


np.save('./res_dict/metric_dict_iter'+app+'.npy', metric_dict)
print('./res_dict/metric_dict_iter'+app+'.npy')

metric_dict = np.load('./res_dict/metric_dict_iter'+app+'.npy', allow_pickle=True)
metric_dict = metric_dict.item()

get_schsisnr = np.concatenate(metric_dict['sch_sisnr'], axis = 0)
print(get_schsisnr.shape)
get_schsnr = np.concatenate(metric_dict['sch_snr'], axis = 0)
print(get_schsnr.shape)
get_schsdr = np.concatenate(metric_dict['sch_sdr'], axis = 0)
print(get_schsdr.shape)
get_t60 = np.concatenate(metric_dict['t60'], axis = 0)
print(get_t60.shape)
get_spatsep = np.concatenate(metric_dict['spat_sep'], axis = 0)
print(get_spatsep.shape)
get_snr_frame = np.concatenate(metric_dict['snr_frame'], axis = 0)
print(get_snr_frame.shape)

print(np.mean(get_schsisnr))
print(np.mean(get_schsnr))
print(np.mean(get_schsdr))

def get_xy(val_x, val_y, prec = 100):
    y_x = dict()
    for batch_idx in range(val_x.shape[0]):
        for iter_idx in range(val_x.shape[1]):
            x_idx = int(val_x[batch_idx, iter_idx] * prec)
            if not x_idx in y_x:
                y_x[x_idx] = []
            y_x[x_idx].append(val_y[batch_idx, iter_idx])

    xs, ys = [], []
    # print(res_dict['pow'])
    for xx in y_x:
        xs.append(xx / prec)
        ys.append(np.mean(np.array(y_x[xx])))
    # print(t60s)
    idx = [i[0] for i in sorted(enumerate(xs), key=lambda x:x[1])]#[5:]
    xs = np.array(xs)[idx]
    ys = np.array(ys)[idx]
    return xs, ys, y_x

def smooth(xs, ys):
    # return (np.array([np.mean(xs[i-10:i + 10]) for i in range(xs.shape[0])]),
    #         np.array([np.mean(ys[i-10:i + 10]) for i in range(xs.shape[0])]))
    return (np.array([np.mean(xs[i-20:i + 20]) for i in range(xs.shape[0])]),
            np.array([np.mean(ys[i-20:i + 20]) for i in range(xs.shape[0])]))

plt.figure()
xs, ys, _ = get_xy(get_spatsep, get_snr_frame, prec = 1)
xs, ys = smooth(xs, ys)
plt.plot(xs, ys, label = 'MFTS')
plt.xlabel('Angular separation [°]')
plt.ylabel('SNR [dB]')
plt.legend()
plt.savefig('./figs/trans_snr-spatsep'+app+'.png')