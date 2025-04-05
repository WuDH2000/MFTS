from src.training.stinfor_dataset import MFTSDataset as infer_dataset
import src.models.dcsa as network_wl
import src.models.spatialnet.mamba_spatinfor as network_sts
import src.models.spatialnet.mamba_noinfor_env_getpoints as network_rs
import src.models.spatialnet.mamba_spatinfor_bf as network_bf
import src.models.spatialnet.mamba_envinfor as network_ef
# import src.models.transversion_dcsa as network_wl
# import src.models.spatialnet.transversion_mamba_spatinfor as network_sts
# import src.models.spatialnet.transversion_mamba_noinfor_env_getpoints as network_rs
# import src.models.spatialnet.transversion_mamba_spatinfor_bf as network_bf
import multiprocessing
import argparse
import torch
from src.helpers import utils
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from math import factorial
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional import signal_noise_ratio as snr
from torchmetrics.functional import signal_distortion_ratio as sdr
from src.training.env_utils import get_env_points, get_env_by_inpterp


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(os.getpid())

app = '_mfts_nsrc2_37_frame_bg2'
print(app)
loc_onsep = False
max_iter = 3
sel_index = 0
metric_dict = dict()

metric_dict['wadp'] = []
metric_dict['wadt'] = []
metric_dict['nmse'] = []

metric_dict['sisnr'] = []
metric_dict['snr'] = []
metric_dict['sdr'] = []
metric_dict['sisnr0'] = []
metric_dict['snr0'] = []
metric_dict['sdr0'] = []
metric_dict['sch_sisnr'] = []
metric_dict['sch_snr'] = []
metric_dict['sch_sdr'] = []

metric_dict['num_srcs'] = []
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
num_workers = min(multiprocessing.cpu_count(), args.n_workers)
kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    }
data_test = infer_dataset(**args.test_data)
test_loader = torch.utils.data.DataLoader(data_test,
                                               batch_size=16,
                                               shuffle=False, **kwargs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######sep_env
model_rs = network_rs.Net(dim_output = 2 * 2).to(device)
model_rs_path = 'experiments/trans_env/mamba_32_bgnoise_fix2src/45.pt'
# model_rs_path = 'experiments/trans_env/mamba_32_bgnoise_3src/77.pt'
# model_rs_path = 'experiments/spatialnet_noinfor_env_linear_128_nmseloss_mp/32.pt'
model_rs.load_state_dict(
    torch.load(model_rs_path, map_location=device)["model_state_dict"]
)

###
model_ml = network_wl.Net().to(device)
model_ml_path = 'experiments/trans_initloc/dcsa_2src/86.pt'
# model_ml_path = 'experiments/trans_initloc/dcsa_3src/51.pt'
# model_ml_path = 'experiments/waveformer_loc_onmix_catenv_mp_diff/20.pt'
model_ml.load_state_dict(
        torch.load(model_ml_path, map_location=device)["model_state_dict"]
    )

#########spatinfor sep
model_sts = network_sts.Net().to(device)
model_sts_path = 'experiments/trans_STIV/mamba_32_bgnoise_fix2src/56.pt'
# model_sts_path = "experiments/trans_STIV/mamba_32_bgnoise_3src/45.pt"
# model_sts_path = 'experiments/spatialnet_multraj_IV_128_linear_snrloss/24.pt'
model_sts.load_state_dict(
        torch.load(model_sts_path, map_location=device)["model_state_dict"]
)

######### bf
model_bf = network_bf.Net().to(device)
model_bf_path = 'experiments/trans_bf/bf_bgnoise_maxsrc2/37.pt'
# model_bf_path = "experiments/trans_bf/bf_bgnoise_maxsrc3/15.pt"
# model_bf_path = 'experiments/spatialnet_bf_oniter_IV_mp_catmix_snrloss_gradtrack/27.pt'
model_bf.load_state_dict(
        torch.load(model_bf_path, map_location=device)["model_state_dict"]
)

##########wavinfor loc
model_wl = network_wl.Net(loc_onsing = False).to(device)
model_wl_path = 'experiments/trans_precloc/dcsa_precloc_nsrc2/32.pt'
# model_wl_path = 'experiments/trans_precloc/dcsa_precloc/38.pt'
# model_wl_path = 'experiments/loc_onsepplusenv_catmix_onlabel_snrlosssep_grad/12.pt'
model_wl.load_state_dict(
        torch.load(model_wl_path, map_location=device)["model_state_dict"]
    )


def nmse(pred, tgt):
    #[B, s, 48000]
    #[s, 48000]
    return 10 * torch.log10(torch.sum((tgt - pred) ** 2, axis = -1) / torch.sum(tgt ** 2, axis = -1))

def ener_weighted_locloss(pred, tgt, est_env):
    getloss = torch.mean((pred.permute(0, 2, 1) * est_env 
                          - tgt.permute(0, 2, 1) * est_env) ** 2, axis = (-1, -2))
    return getloss


model_rs.eval()
model_ml.eval()
model_sts.eval()
model_wl.eval()
model_bf.eval()

with torch.no_grad():
    for batch_idx, (mixed, clean_wav, traj_t, tgt_env_points, T60) in enumerate(tqdm(test_loader, desc='Test', ncols=100)):
        rec_adp = []
        rec_adt = []
        rec_snr = []
        rec_sdr = []
        rec_sisnr = []
        rec_snr0 = []
        rec_sdr0 = []
        rec_sisnr0 = []
        rec_schsnr = []
        rec_schsdr = []
        rec_schsisnr = []
        rec_nmse = []
        rec_numsrcs = []
        
        rec_t60 = []
        rec_spatsep = []
        
        mixed = mixed.to(device) #[B, 4, 48000]
        clean_wav = clean_wav.to(device)  #[B, ns, 4, 48000]
        traj_t = traj_t.to(device) #[B, ns, 48000, 3]  # to calculate loss
        tgt_env_points = tgt_env_points.to(device) #[B, maxns, T]
        
        traj_sph = network_wl.cart2sph(traj_t) #[B, ns, len, 3]
        spat_sep = network_wl.angular_error(
                    traj_sph[:, 0, :, 1], 
                    traj_sph[:, 0, :, 2],
                    traj_sph[:, 1, :, 1],
                    traj_sph[:, 1, :, 2]) / torch.pi * 180 #[B, len]

        with torch.no_grad():
            output = model_rs(mixed.detach())  #[B, ns, T]
            # print(output.shape)
            # print(tgt_env_points.shape)
            _, est_env_points = network_rs.loss(output, tgt_env_points, return_est=True)
            est_env_points_s = []
            tgt_env_points_s = []
            label_traj_s = []
            mixed_s = []
            clean_wav_s = []
            
            for bb in range(mixed.shape[0]):
                est_env_points_s_b = est_env_points[bb] #[maxns, T]
                tgt_env_points_s_b = tgt_env_points[bb] #[maxns, T]
                label_traj_s_b = traj_t[bb] #[maxns, len, 3]
                clean_wav_s_b = clean_wav[bb] #[maxns, 4, 48000]
                # print(label_traj_s_b.shape)
                rec_numsrcs.append(np.array(torch.sum((torch.max(tgt_env_points_s_b, axis = -1)[0] > 0.25).int()).item()).reshape(1))
                exist_tensor = ((torch.max(est_env_points_s_b, axis = -1)[0] > 0.25) 
                                        & (torch.max(tgt_env_points_s_b, axis = -1)[0] > 0.25))
                    # print(exist_tensor.shape)
                est_env_points_s_b = est_env_points_s_b[exist_tensor]
                tgt_env_points_s_b = tgt_env_points_s_b[exist_tensor]
                label_traj_s_b = label_traj_s_b[exist_tensor]
                clean_wav_s_b = clean_wav_s_b[exist_tensor]
                
                num_directions = est_env_points_s_b.shape[0]
                if num_directions == 0:
                    continue
                sel_idx = np.random.randint(0, num_directions)
                
                est_env_points_s_b = est_env_points_s_b[sel_idx:sel_idx + 1] #[1, T]
                tgt_env_points_s_b = tgt_env_points_s_b[sel_idx:sel_idx + 1]
                label_traj_s_b = label_traj_s_b[sel_idx]   #[len, 3]
                clean_wav_s_b = clean_wav_s_b[sel_idx]
                
                est_env_points_s.append(est_env_points_s_b.unsqueeze(0))
                tgt_env_points_s.append(tgt_env_points_s_b.unsqueeze(0))
                label_traj_s.append(label_traj_s_b.unsqueeze(0))
                mixed_s.append(mixed[bb].unsqueeze(0))
                clean_wav_s.append(clean_wav_s_b.unsqueeze(0))
                
                rec_t60.append(T60[bb].cpu().detach().numpy().reshape(1, 1))
                rec_spatsep.append(spat_sep[bb].unsqueeze(0))
            
            est_env_points_s = torch.cat(est_env_points_s, axis = 0) #[B, 1, T]
            tgt_env_points_s = torch.cat(tgt_env_points_s, axis = 0)
            label_traj_s = torch.cat(label_traj_s, axis = 0)  #[B, len, 3]
            mixed_s = torch.cat(mixed_s, axis = 0)
            clean_wav_s = torch.cat(clean_wav_s, axis = 0)  #[B, 4, 48000]
            rec_spatsep = torch.cat(rec_spatsep, axis = 0)  #[B, 48000]

            est_env_s = get_env_by_inpterp(est_env_points_s) #[B, 1, len]
            tgt_env_s = get_env_by_inpterp(tgt_env_points_s)
            
            get_nmse = nmse(est_env_points_s, tgt_env_points_s) #[B, 1]
            print('nmse', get_nmse.cpu().detach().numpy())
            rec_nmse.append(get_nmse.reshape(-1, 1).cpu().detach().numpy())

            loc_in = torch.cat([mixed_s, est_env_s], axis = 1) #
                
            est_traj = model_ml(loc_in.detach()) #[B, T, 3]
            loc_metrics = network_wl.metrics(est_traj, label_traj_s, est_env_s, tgt_env_s, keepbatch = True)
            rec_adp.append(loc_metrics['adp'][0].reshape(-1, 1))
            rec_adt.append(loc_metrics['adt'][0].reshape(-1, 1))
            print(loc_metrics)
            
            est_traj_inf = est_traj #[B, 48000, 3]
            
            for iter_ in range(max_iter):
                traj_tf = torch.cat([torch.mean(est_traj_inf[:, i:i + 256], dim = 1, keepdim = True) 
                        for i in range(0, est_traj_inf.shape[1] - 256 + 1, 128)], axis = 1)
                # print(spat_inf.shape)
                traj_tf = torch.cat([torch.mean(est_traj_inf[:, :128], dim = 1, keepdim = True),
                                traj_tf, 
                                torch.mean(est_traj_inf[:, -128:], dim = 1, keepdim = True)], axis = 1) #[B, T, 3]
                traj_tf = traj_tf.permute(0, 2, 1)  #[B, 3, T]
                traj_tf = traj_tf * est_env_points_s  #[B, 3, T]
                # Run through the model
                est_clean_wav = model_sts([mixed_s, traj_tf.detach()]) #[B, 4, 48000]
                get_snr = snr(est_clean_wav, clean_wav_s).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
                rec_snr.append(get_snr)
                print('snr', get_snr)
                get_sdr = sdr(est_clean_wav, clean_wav_s).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
                rec_sdr.append(get_sdr)
                print('sdr', get_sdr)
                get_sisnr = si_snr(est_clean_wav, clean_wav_s).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
                rec_sisnr.append(get_sisnr)
                print('sisnr', get_sisnr)
                
                get_snr = snr(est_clean_wav[:, :1], clean_wav_s[:, :1]).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
                rec_snr0.append(get_snr)
                print('snr0', get_snr)
                get_sdr = sdr(est_clean_wav[:, :1], clean_wav_s[:, :1]).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
                rec_sdr0.append(get_sdr)
                print('sdr0', get_sdr)
                get_sisnr = si_snr(est_clean_wav[:, :1], clean_wav_s[:, :1]).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
                rec_sisnr0.append(get_sisnr)
                print('sisnr0', get_sisnr)
                
                output_wl = est_clean_wav / torch.max(torch.max(torch.abs(est_clean_wav),
                        axis = -1)[0], axis = -1)[0].unsqueeze(-1).unsqueeze(-1)
                input_loc = torch.cat([output_wl, mixed_s, est_env_s], axis = 1)
                est_traj = model_wl(input_loc.detach()) #[B, t, 3]
                loc_metrics = network_wl.metrics(est_traj, label_traj_s, est_env_s, tgt_env_s, keepbatch = True)
                rec_adp.append(loc_metrics['adp'][0].reshape(-1, 1))
                rec_adt.append(loc_metrics['adt'][0].reshape(-1, 1))
                print(loc_metrics)
                
                est_traj_inf = est_traj
                    
                if iter_ == 1:
                    traj_tf = torch.cat([torch.mean(est_traj_inf[:, i:i + 256], dim = 1, keepdim = True) 
                            for i in range(0, est_traj_inf.shape[1] - 256 + 1, 128)], axis = 1)
                    # print(spat_inf.shape)
                    traj_tf = torch.cat([torch.mean(est_traj_inf[:, :128], dim = 1, keepdim = True),
                                    traj_tf, 
                                    torch.mean(est_traj_inf[:, -128:], dim = 1, keepdim = True)], axis = 1) #[B, T, 3]
                    traj_tf = traj_tf.permute(0, 2, 1)  #[B, 3, T]
                    # print(traj_tf.shape)
                    # print(est_env_points_s.shape)
                    traj_tf = traj_tf * est_env_points_s  #[B, 3, T]
                    bf_wav = torch.cat([est_clean_wav.detach(), mixed_s.detach()], axis = 1) #[B, 8, tt]
                    est_sch_wav = model_bf([bf_wav.detach(), traj_tf.detach()])
                    
                    get_snr = snr(est_sch_wav[:, :1], clean_wav_s[:, :1]).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
                    rec_schsnr.append(get_snr)
                    print('schsnr', get_snr)
                    get_sdr = sdr(est_sch_wav[:, :1], clean_wav_s[:, :1]).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
                    rec_schsdr.append(get_sdr)
                    print('schsdr', get_sdr)
                    get_sisnr = si_snr(est_sch_wav[:, :1], clean_wav_s[:, :1]).mean(dim = -1).reshape(-1, 1).cpu().detach().numpy()
                    rec_schsisnr.append(get_sisnr)
                    
            B = est_sch_wav.shape[0]
            est_sch_wav_frame = est_sch_wav[:, 0].reshape(B, 125, -1)
            clean_wav_frame = clean_wav_s[:, 0].reshape(B, 125, -1)
            rec_spatsep = torch.mean(rec_spatsep.reshape(B, 125, -1), axis = -1)
            snr_frame = snr(est_sch_wav_frame, clean_wav_frame) #[B, 125]

        metric_dict['wadp'].append(np.concatenate(rec_adp, axis = -1))
        metric_dict['wadt'].append(np.concatenate(rec_adt, axis = -1))
        metric_dict['nmse'].append(np.concatenate(rec_nmse, axis = -1))
        metric_dict['sisnr'].append(np.concatenate(rec_sisnr, axis = -1))
        metric_dict['snr'].append(np.concatenate(rec_snr, axis = -1))
        metric_dict['sdr'].append(np.concatenate(rec_sdr, axis = -1))
        metric_dict['sisnr0'].append(np.concatenate(rec_sisnr0, axis = -1))
        metric_dict['snr0'].append(np.concatenate(rec_snr0, axis = -1))
        metric_dict['sdr0'].append(np.concatenate(rec_sdr0, axis = -1))
        metric_dict['sch_sisnr'].append(np.concatenate(rec_schsisnr, axis = -1))
        metric_dict['sch_snr'].append(np.concatenate(rec_schsnr, axis = -1))
        metric_dict['sch_sdr'].append(np.concatenate(rec_schsdr, axis = -1))
        metric_dict['num_srcs'].append(np.concatenate(rec_numsrcs, axis = -1))
        
        metric_dict['t60'].append(np.concatenate(rec_t60, axis = 0))
        metric_dict['spat_sep'].append(rec_spatsep.cpu().detach().numpy())
        metric_dict['snr_frame'].append(snr_frame.cpu().detach().numpy())
        # if (batch_idx + 1) % 3 == 0:
        #     break
np.save('./res_dict/metric_dict_iter'+app+'.npy', metric_dict)
print('./res_dict/metric_dict_iter'+app+'.npy')

metric_dict = np.load('./res_dict/metric_dict_iter'+app+'.npy', allow_pickle=True)
metric_dict = metric_dict.item()

get_wadp = np.concatenate(metric_dict['wadp'], axis = 0)
print(get_wadp.shape)
get_wadt = np.concatenate(metric_dict['wadt'], axis = 0)
print(get_wadt.shape)
get_nmse = np.concatenate(metric_dict['nmse'], axis = 0)
print(get_nmse.shape)
get_sisnr = np.concatenate(metric_dict['sisnr'], axis = 0)
print(get_sisnr.shape)
get_snr = np.concatenate(metric_dict['snr'], axis = 0)
print(get_snr.shape)
get_sdr = np.concatenate(metric_dict['sdr'], axis = 0)
print(get_sdr.shape)
get_sisnr0 = np.concatenate(metric_dict['sisnr0'], axis = 0)
print(get_sisnr0.shape)
get_snr0 = np.concatenate(metric_dict['snr0'], axis = 0)
print(get_snr0.shape)
get_sdr0 = np.concatenate(metric_dict['sdr0'], axis = 0)
print(get_sdr0.shape)
get_schsisnr = np.concatenate(metric_dict['sch_sisnr'], axis = 0)
print(get_sisnr0.shape)
get_schsnr = np.concatenate(metric_dict['sch_snr'], axis = 0)
print(get_schsnr.shape)
get_schsdr = np.concatenate(metric_dict['sch_sdr'], axis = 0)
print(get_schsdr.shape)
get_numsrcs = np.concatenate(metric_dict['num_srcs'], axis = 0)
print(get_numsrcs.shape)
get_t60 = np.concatenate(metric_dict['t60'], axis = 0)
print(get_t60.shape)

get_spatsep = np.concatenate(metric_dict['spat_sep'], axis = 0)
print(get_spatsep.shape)
get_snrframe = np.concatenate(metric_dict['snr_frame'], axis = 0)
print(get_snrframe.shape)


print('average')
for num_iter in range(get_sdr.shape[-1]):
    print('wadp{}'.format(num_iter), np.mean(get_wadp[:, num_iter]))
    print('wadt{}'.format(num_iter), np.mean(get_wadt[:, num_iter]))
    print('sdr-{}'.format(num_iter), np.mean(get_sdr[:, num_iter]))
    print('snr-{}'.format(num_iter), np.mean(get_snr[:, num_iter]))
    print('sisnr-{}'.format(num_iter), np.mean(get_sisnr[:, num_iter]))
    print('sdr0-{}'.format(num_iter), np.mean(get_sdr0[:, num_iter]))
    print('snr0-{}'.format(num_iter), np.mean(get_snr0[:, num_iter]))
    print('sisnr0-{}'.format(num_iter), np.mean(get_sisnr0[:, num_iter]))
print('wadp{}'.format(get_sdr.shape[-1]), np.mean(get_wadp[:, -1]))
print('wadt{}'.format(get_sdr.shape[-1]), np.mean(get_wadt[:, -1]))
print('schsdr-{}'.format(num_iter), np.mean(get_schsdr[:, -1]))
print('schsnr-{}'.format(num_iter), np.mean(get_schsnr[:, -1]))
print('schsisnr-{}'.format(num_iter), np.mean(get_schsisnr[:, -1]))

# print('src2')
# get_wadp_2 = get_wadp[get_numsrcs == 2]
# get_wadt_2 = get_wadt[get_numsrcs == 2]
# get_sisnr_2 = get_sisnr[get_numsrcs == 2]
# get_snr_2 = get_snr[get_numsrcs == 2]
# get_sdr_2 = get_sdr[get_numsrcs == 2]
# get_sisnr0_2 = get_sisnr0[get_numsrcs == 2]
# get_snr0_2 = get_snr0[get_numsrcs == 2]
# get_sdr0_2 = get_sdr0[get_numsrcs == 2]
# get_schsisnr_2 = get_schsisnr[get_numsrcs == 2]
# get_schsnr_2 = get_schsnr[get_numsrcs == 2]
# get_schsdr_2 = get_schsdr[get_numsrcs == 2]

# for num_iter in range(get_sdr.shape[-1]):
#     print('wadp{}'.format(num_iter), np.mean(get_wadp_2[:, num_iter]))
#     print('wadt{}'.format(num_iter), np.mean(get_wadt_2[:, num_iter]))
#     print('sdr-{}'.format(num_iter), np.mean(get_sdr_2[:, num_iter]))
#     print('snr-{}'.format(num_iter), np.mean(get_snr_2[:, num_iter]))
#     print('sisnr-{}'.format(num_iter), np.mean(get_sisnr_2[:, num_iter]))
#     print('sdr0-{}'.format(num_iter), np.mean(get_sdr0_2[:, num_iter]))
#     print('snr0-{}'.format(num_iter), np.mean(get_snr0_2[:, num_iter]))
#     print('sisnr0-{}'.format(num_iter), np.mean(get_sisnr0_2[:, num_iter]))
# print('wadp{}'.format(get_sdr.shape[-1]), np.mean(get_wadp_2[:, -1]))
# print('wadt{}'.format(get_sdr.shape[-1]), np.mean(get_wadt_2[:, -1]))
# print('schsdr-{}'.format(num_iter), np.mean(get_schsdr_2[:, -1]))
# print('schsnr-{}'.format(num_iter), np.mean(get_schsnr_2[:, -1]))
# print('schsisnr-{}'.format(num_iter), np.mean(get_schsisnr_2[:, -1]))

# print('src3')
# get_wadp_3 = get_wadp[get_numsrcs == 3]
# get_wadt_3 = get_wadt[get_numsrcs == 3]
# get_sisnr_3 = get_sisnr[get_numsrcs == 3]
# get_snr_3 = get_snr[get_numsrcs == 3]
# get_sdr_3 = get_sdr[get_numsrcs == 3]
# get_sisnr0_3 = get_sisnr0[get_numsrcs == 3]
# get_snr0_3 = get_snr0[get_numsrcs == 3]
# get_sdr0_3 = get_sdr0[get_numsrcs == 3]
# get_schsisnr_3 = get_schsisnr[get_numsrcs == 3]
# get_schsnr_3 = get_schsnr[get_numsrcs == 3]
# get_schsdr_3 = get_schsdr[get_numsrcs == 3]

# for num_iter in range(get_sdr.shape[-1]):
#     print('wadp{}'.format(num_iter), np.mean(get_wadp_3[:, num_iter]))
#     print('wadt{}'.format(num_iter), np.mean(get_wadt_3[:, num_iter]))
#     print('sdr-{}'.format(num_iter), np.mean(get_sdr_3[:, num_iter]))
#     print('snr-{}'.format(num_iter), np.mean(get_snr_3[:, num_iter]))
#     print('sisnr-{}'.format(num_iter), np.mean(get_sisnr_3[:, num_iter]))
#     print('sdr0-{}'.format(num_iter), np.mean(get_sdr0_3[:, num_iter]))
#     print('snr0-{}'.format(num_iter), np.mean(get_snr0_3[:, num_iter]))
#     print('sisnr0-{}'.format(num_iter), np.mean(get_sisnr0_3[:, num_iter]))
# print('wadp{}'.format(get_sdr.shape[-1]), np.mean(get_wadp_3[:, -1]))
# print('wadt{}'.format(get_sdr.shape[-1]), np.mean(get_wadt_3[:, -1]))
# print('schsdr-{}'.format(num_iter), np.mean(get_schsdr_3[:, -1]))
# print('schsnr-{}'.format(num_iter), np.mean(get_schsnr_3[:, -1]))
# print('schsisnr-{}'.format(num_iter), np.mean(get_schsisnr_3[:, -1]))

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
xs, ys, _ = get_xy(get_spatsep, get_snrframe, prec = 1)
xs, ys = smooth(xs, ys)
plt.plot(xs, ys, label = 'MFTS')
plt.xlabel('Angular separation [°]')
plt.ylabel('SNR [dB]')
plt.legend()
plt.savefig('./figs/trans_snr-spatsep'+app+'.png')