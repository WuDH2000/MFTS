from src.training.stinfor_dataset import MFTSDataset as infer_dataset
import src.models.spatialnet.mamba_noinfor_env_getpoints as network_rs
import src.models.dcsa as network_ml
import src.models.dcsa_pit as network_rl
import argparse
import torch
from src.helpers import utils
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from math import factorial
import scipy.special as sp
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional import signal_noise_ratio as snr
from torchmetrics.functional import signal_distortion_ratio as sdr
from src.training.traj_utils.sph import sph2cart, cart2sph, angular_error, mvdr_sh_freq
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(os.getpid())

app = '_trans_loc_abla'
# print(app)

# metric_dict = dict()
# metric_dict['t60'] = []
# metric_dict['wadt_wl'] = []
# metric_dict['wadt_rl'] = []
# metric_dict['adwl_samp'] = []
# metric_dict['adrl_samp'] = []
# metric_dict['env_samp'] = []

# parser = argparse.ArgumentParser()
# exp_dir = "experiments/invest"
# args = parser.parse_args()
# params = utils.Params(os.path.join(exp_dir, 'config.json'))
# for k, v in params.__dict__.items():
#     vars(args)[k] = v
# use_cuda = torch.cuda.is_available()
# kwargs = {
#         'num_workers': 4,
#         'pin_memory': True
#     }
# data_test = infer_dataset(**args.test_data)
# test_loader = torch.utils.data.DataLoader(data_test,
#                                                batch_size=16,
#                                                shuffle=False, **kwargs)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #######sep_env
# model_rs = network_rs.Net(dim_output = 2 * 2).to(device)
# # model_rs_path = 'experiments/trans_env/mamba_32_bgnoise_3src/77.pt'
# model_rs_path = 'experiments/trans_env/mamba_32_bgnoise_fix2src/45.pt'
# model_rs.load_state_dict(
#     torch.load(model_rs_path, map_location=device)["model_state_dict"]
# )

# model_ml = network_ml.Net().to(device)
# # model_ml_path = 'experiments/trans_initloc/dcsa_3src/51.pt'
# model_ml_path = 'experiments/trans_initloc/dcsa_2src/86.pt'
# model_ml.load_state_dict(
#     torch.load(model_ml_path, map_location=device)["model_state_dict"]
# )

# model_rl = network_rl.Net().to(device)
# model_rl_path = 'experiments/trans_noinforloc/dcsa_2src_diffloss_pit/86.pt'
# model_rl.load_state_dict(
#     torch.load(model_rl_path, map_location = device)['model_state_dict']
# )

# def nmse(pred, tgt):
#     #[B, s, 48000]
#     #[s, 48000]
#     return 10 * torch.log10(torch.sum((tgt - pred) ** 2, axis = -1) / torch.sum(tgt ** 2, axis = -1))

# max_filter = torch.nn.MaxPool1d(kernel_size=256, stride=128).cuda()
# def get_env_points(wav):
#     res =  max_filter(torch.abs(wav))
#     return res
# def get_env_by_inpterp(points, length = 48000):
#     sq_flag = False
#     if len(points.shape) == 4:
#         sq_flag = True
#         points = points.squeeze(2)
#     res =  torch.nn.functional.interpolate(points, (length), mode = 'linear') #[B, 2, 48000]
#     if sq_flag:
#         res = res.unsqueeze(2)
#     return res
# def get_env(wav):
#     res =  max_filter(torch.abs(wav))
#     return torch.nn.functional.interpolate(res, (wav.shape[-1]), mode = 'linear')

# model_rs.eval()
# model_ml.eval()

# with torch.no_grad():
#     for batch_idx, (mixed, clean_wav, traj_t, tgt_env_points, T60) in enumerate(tqdm(test_loader, desc='Test', ncols=100)):
#         rec_adt_wl = []
#         rec_adt_rl = []
#         rec_t60 = []
        
#         mixed = mixed.to(device) #[B, 4, 48000]
#         clean_wav = clean_wav.to(device)  #[B, ns, 4, 48000]
#         traj_t = traj_t.to(device) #[B, ns, 48000, 3]  # to calculate loss
#         tgt_env_points = tgt_env_points.to(device) #[B, maxns, T]
        
#         est_traj_rl = model_rl(mixed)
#         _, est_traj_rl = network_rl.loss(est_traj_rl, traj_t, return_est=True)

#         with torch.no_grad():
#             output = model_rs(mixed.detach())  #[B, ns, T]
#             # print(output.shape)
#             # print(tgt_env_points.shape)
#             _, est_env_points = network_rs.loss(output, tgt_env_points, return_est=True)
#             est_env_points_s = []
#             tgt_env_points_s = []
#             label_traj_s = []
#             mixed_s = []
#             clean_wav_s = []
#             est_traj_rl_s = []
#             rec_t60 = []
            
#             for bb in range(mixed.shape[0]):
#                 # print(num_directions[bb])
#                 est_env_points_s_b = est_env_points[bb] #[maxns, T]
#                 tgt_env_points_s_b = tgt_env_points[bb] #[maxns, T]
#                 label_traj_s_b = traj_t[bb] #[maxns, len, 3]
#                 clean_wav_s_b = clean_wav[bb] #[maxns, 4, 48000]
#                 est_traj_rl_s_b = est_traj_rl[bb] #[maxns, len, 3]
#                 exist_tensor = ((torch.max(est_env_points_s_b, axis = -1)[0] > 0.25) 
#                                         & (torch.max(tgt_env_points_s_b, axis = -1)[0] > 0.25))
#                     # print(exist_tensor.shape)
#                 est_env_points_s_b = est_env_points_s_b[exist_tensor]
#                 tgt_env_points_s_b = tgt_env_points_s_b[exist_tensor]
#                 label_traj_s_b = label_traj_s_b[exist_tensor]
#                 clean_wav_s_b = clean_wav_s_b[exist_tensor]
#                 est_traj_rl_s_b = est_traj_rl_s_b[exist_tensor]
                    
#                 num_directions = est_env_points_s_b.shape[0]
#                 if num_directions == 0:
#                     continue
#                 sel_idx = np.random.randint(0, num_directions)
                    
#                 est_env_points_s_b = est_env_points_s_b[sel_idx:sel_idx + 1] #[1, T]
#                 tgt_env_points_s_b = tgt_env_points_s_b[sel_idx:sel_idx + 1]
#                 label_traj_s_b = label_traj_s_b[sel_idx]   #[len, 3]
#                 clean_wav_s_b = clean_wav_s_b[sel_idx]
#                 est_traj_rl_s_b = est_traj_rl_s_b[sel_idx]
                
#                 est_env_points_s.append(est_env_points_s_b.unsqueeze(0))
#                 tgt_env_points_s.append(tgt_env_points_s_b.unsqueeze(0))
#                 label_traj_s.append(label_traj_s_b.unsqueeze(0))
#                 mixed_s.append(mixed[bb].unsqueeze(0))
#                 clean_wav_s.append(clean_wav_s_b.unsqueeze(0))
#                 est_traj_rl_s.append(est_traj_rl_s_b.unsqueeze(0))
#                 rec_t60.append(T60[bb].cpu().detach().numpy())
            
#             est_env_points_s = torch.cat(est_env_points_s, axis = 0) #[B, 1, T]
#             tgt_env_points_s = torch.cat(tgt_env_points_s, axis = 0)
#             label_traj_s = torch.cat(label_traj_s, axis = 0)  #[B, len, 3]
#             mixed_s = torch.cat(mixed_s, axis = 0)
#             clean_wav_s = torch.cat(clean_wav_s, axis = 0)  #[B, 4, 48000]
#             est_traj_rl_s = torch.cat(est_traj_rl_s, axis = 0) #[B, len, 3]
#             rec_t60 = np.array(rec_t60).reshape(-1, 1)
                    
#             est_env_s = get_env_by_inpterp(est_env_points_s) #[B, 1, len]
#             tgt_env_s = get_env_by_inpterp(tgt_env_points_s)

#             loc_in = torch.cat([mixed_s, est_env_s], axis = 1) #
                
#             est_traj_wl = model_ml(loc_in.detach()) #[B, T, 3]
#             loc_metrics = network_ml.metrics(est_traj_wl, label_traj_s, 
#                                              est_env_s, tgt_env_s, keepbatch = True)
#             rec_adt_wl.append(loc_metrics['adt'][0].reshape(-1, 1))
#             print(loc_metrics)
            
#             loc_metrics = network_ml.metrics(est_traj_rl_s, label_traj_s, 
#                                              est_env_s, tgt_env_s, keepbatch = True)
#             rec_adt_rl.append(loc_metrics['adt'][0].reshape(-1, 1))
#             print(loc_metrics)
            
#         metric_dict['t60'].append(rec_t60)
#         metric_dict['wadt_rl'].append(np.concatenate(rec_adt_rl, axis = -1))
#         metric_dict['wadt_wl'].append(np.concatenate(rec_adt_wl, axis = -1))
        
#         est_traj_rl_s_sph = cart2sph(est_traj_rl_s) #[B, len, 3]
#         est_traj_wl_sph = cart2sph(est_traj_wl)  #[B, len, 3]
#         label_traj_s_sph = cart2sph(label_traj_s) #[B, len, 3]
#         ad_rl = angular_error(est_traj_rl_s_sph[..., 1], 
#                 est_traj_rl_s_sph[..., 2], label_traj_s_sph[..., 1],
#                 label_traj_s_sph[..., 2]) / torch.pi * 180 #[B, len]
#         ad_wl = angular_error(est_traj_wl_sph[..., 1], 
#                 est_traj_wl_sph[..., 2], label_traj_s_sph[..., 1],
#                 label_traj_s_sph[..., 2]) / torch.pi * 180
#         print('adrl', ad_rl.mean())
#         print('adwl', ad_wl.mean())
#         get_tpow = tgt_env_s[:, 0]  #[B, len]
        
#         metric_dict['env_samp'].append(get_tpow[:, ::100].cpu().detach().numpy())
#         metric_dict['adrl_samp'].append(ad_rl[:, ::100].cpu().detach().numpy())
#         metric_dict['adwl_samp'].append(ad_wl[:, ::100].cpu().detach().numpy())
        
#         # if (batch_idx + 1) % 3 == 0:
#         #     break

# np.save('./res_dict/metric_dict_iter'+app+'.npy', metric_dict)
# print('./res_dict/metric_dict_iter'+app+'.npy')

metric_dict = np.load('./res_dict/metric_dict_iter'+app+'.npy', allow_pickle=True)
metric_dict = metric_dict.item()

env_samp = np.concatenate(metric_dict['env_samp'], axis = 0)
print(env_samp.shape)
adrl_samp = np.concatenate(metric_dict['adrl_samp'], axis = 0)
print(adrl_samp.shape)
adwl_samp = np.concatenate(metric_dict['adwl_samp'], axis = 0)
print(adwl_samp.shape)
t60 = np.concatenate(metric_dict['t60'], axis = 0)
print(t60.shape)
wadt_rl = np.concatenate(metric_dict['wadt_rl'], axis = 0)
print(wadt_rl.shape)
wadt_wl = np.concatenate(metric_dict['wadt_wl'], axis = 0)
print(wadt_wl.shape)


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
    stm = 5
    ena = 5
    return (np.array([np.mean(xs[max(i-stm, 0):min(i + ena, xs.shape[0])]) 
                      for i in range(xs.shape[0])]),
            np.array([np.mean(ys[max(i-stm, 0):min(i + ena, ys.shape[0])]) 
                      for i in range(xs.shape[0])]))


plt.figure()
xs, ys, _ = get_xy(t60, wadt_rl, prec = 100)
xs, ys = smooth(xs, ys)
plt.plot(xs, ys, label = 'mix-track')
xs, ys, _ = get_xy(t60, wadt_wl, prec = 100)
xs, ys = smooth(xs, ys)
plt.plot(xs, ys, label = 'init-track')
plt.xlabel('T60 [s]')
plt.ylabel('EWRMSAE [°]')
plt.ylim(15, 30)
plt.legend()
plt.savefig('./figs/trans_rmsae-t60'+app+'.png')



plt.figure()
xs, ys, _ = get_xy(env_samp, adrl_samp, prec = 100)
# xs, ys = smooth(xs, ys)
plt.plot(xs, ys, label = 'mix-track')
xs, ys, _ = get_xy(env_samp, adwl_samp, prec = 100)
# xs, ys = smooth(xs, ys)
plt.plot(xs, ys, label = 'init-track')
plt.xlabel('Amplitude')
plt.ylabel('AE [°]')
plt.legend()
plt.savefig('./figs/trans_ae-amp'+app+'.png')
