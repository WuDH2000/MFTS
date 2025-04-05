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
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional import signal_noise_ratio as snr
from torchmetrics.functional import signal_distortion_ratio as sdr
from src.training.traj_utils.sph import sph2cart, cart2sph, angular_error, mvdr_sh_freq
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(os.getpid())

app = '_trans_estenv_src3'
# print(app)

# metric_dict = dict()
# metric_dict['num_srcs'] = []
# metric_dict['est_srcs'] = []
# metric_dict['nmse'] = []
# metric_dict['acc'] = []
# metric_dict['recall'] = []
# metric_dict['precision'] = []

# parser = argparse.ArgumentParser()
# exp_dir = "experiments/invest"
# args = parser.parse_args()
# params = utils.Params(os.path.join(exp_dir, 'config.json'))
# for k, v in params.__dict__.items():
#     vars(args)[k] = v
# use_cuda = torch.cuda.is_available()
# kwargs = {
#         'num_workers': 0,
#         'pin_memory': True
#     }
# data_test = infer_dataset(**args.test_data)
# test_loader = torch.utils.data.DataLoader(data_test,
#                                                batch_size=1,
#                                                shuffle=False, **kwargs)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #######sep_env
# model_rs = network_rs.Net().to(device)
# model_rs_path = 'experiments/trans_env/mamba_32_bgnoise_3src/77.pt'
# model_rs.load_state_dict(
#     torch.load(model_rs_path, map_location=device)["model_state_dict"]
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

# with torch.no_grad():
#     for batch_idx, (mixed, tgt_env_p) in \
#                     enumerate(tqdm(test_loader, desc='Test', ncols=100)):
        
#         rec_nmse = []
        
#         mixed = mixed.to(device) #[B, 4, 48000]
#         tgt_env_p = tgt_env_p.to(device) #[B, maxns, T]
        
#         est_env_p = model_rs(mixed) #[B, maxns, T]

#         _, est_env_p = network_rs.loss(est_env_p, tgt_env_p, return_est=True)

       
#         tgt_b = tgt_env_p[0] #[ns, T]
#         pred_b = est_env_p[0] #[ns, T]
#         cnt_src_b = 0
#         cnt_src_b_acc = 0
#         rec_nmse_b = 0
        
#         pred_exist = torch.max(pred_b, axis = -1)[0] > 0.25  #[max_ns]
#         tgt_exist = torch.max(tgt_b, axis = -1)[0] > 0.25  #[max_ns]
        
#         get_TP = torch.sum((pred_exist & tgt_exist).int())
#         get_FP = torch.sum((pred_exist & (~tgt_exist)).int())
#         get_FN = torch.sum(((~pred_exist) & tgt_exist).int())
#         get_recall = get_TP / (get_TP + get_FN)
#         get_prec = get_TP / (get_TP + get_FP + (1e-10) * (((get_TP + get_FP) == 0).int()))
#         metric_dict['recall'].append(get_recall.item())
#         metric_dict['precision'].append(get_prec.item())
        
#         tgt_src = torch.sum(tgt_exist.int()).item()
#         pred_src = torch.sum(pred_exist.int()).item()
#         metric_dict['num_srcs'].append(tgt_src)
#         metric_dict['est_srcs'].append(pred_src)
        
#         if torch.sum((tgt_exist == pred_exist).int()) == tgt_b.shape[0]:
#             metric_dict['acc'].append(1)
#         else:
#             metric_dict['acc'].append(0)
        
#         exist_tensor = pred_exist & tgt_exist
        
#         tgt_b_s = tgt_b[exist_tensor]
#         pred_b_s = pred_b[exist_tensor]
#         get_nmse_s = 10 * torch.log10((torch.mean((pred_b_s - tgt_b_s) ** 2, axis = -1))
#                             / (torch.mean(tgt_b_s ** 2, axis = -1)) + 1e-10).mean()
#         metric_dict['nmse'].append(get_nmse_s.item())


# np.save('./res_dict/metric_dict_iter'+app+'.npy', metric_dict)
# print('./res_dict/metric_dict_iter'+app+'.npy')

metric_dict = np.load('./res_dict/metric_dict_iter'+app+'.npy', allow_pickle=True)
metric_dict = metric_dict.item()

get_nmse = np.array(metric_dict['nmse']).reshape(-1, 1)
print(get_nmse.shape)
get_estsrcs = np.array(metric_dict['est_srcs']).reshape(-1, 1)
print(get_estsrcs.shape)
get_tgtsrcs = np.array(metric_dict['num_srcs']).reshape(-1, 1)
print(get_tgtsrcs.shape)
get_acc = np.array(metric_dict['acc']).reshape(-1, 1)
print(get_acc.shape)
get_recall = np.array(metric_dict['recall']).reshape(-1, 1)
print(get_recall.shape)
get_prec = np.array(metric_dict['precision']).reshape(-1, 1)
print(get_prec.shape)

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

x,y, _ = get_xy(get_tgtsrcs, get_nmse)
print(x, y)
x,y, _ = get_xy(get_tgtsrcs, get_acc)
print(x, y)
x,y, _ = get_xy(get_tgtsrcs, get_recall)
print(x, y)
x,y, _ = get_xy(get_tgtsrcs, get_prec)
print(x, y)

def get_stotas(val_x, val_y):
    y_x = dict()
    for batch_idx in range(val_x.shape[0]):
        for iter_idx in range(val_x.shape[1]):
            x_idx = int(val_x[batch_idx, iter_idx])
            if not x_idx in y_x:
                y_x[x_idx] = dict()
            y_idx = val_y[batch_idx, iter_idx]
            if not y_idx in y_x[x_idx]:
                y_x[x_idx][y_idx] = 0
            y_x[x_idx][y_idx] += 1
    return y_x

y_x = get_stotas(get_tgtsrcs, get_estsrcs)
print(y_x)

