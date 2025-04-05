import os
import torch

import numpy as np
import soundfile as sf

from torch.utils import data
import matplotlib.pyplot as plt
import random
from torchmetrics.functional import signal_noise_ratio as snr
from src.training.env_utils import get_env_points, get_env, get_env_by_inpterp
import time
from tqdm import tqdm

class SpkUnkEnvPITDataset(data.Dataset):
    def __init__(self, input_dir, dset, n_src=2, order = 1, hoa=True, if_rev = True,
                 sr = 16000, bg_noise = False, **kwargs):
        self.hoa = hoa
        self.rev = if_rev
        self.data_dir = input_dir
        self.names = os.listdir(self.data_dir + '/pts1')
        self.names = [i for i in self.names if int(i.split('_')[0]) < 60000]
        self.len = len(self.names)
        self.order = order
        print(self.len)
        self.dset = dset
        if dset == 'train':
            self.names = [i for i in self.names if int(i.split('_')[0]) < 50000]
        elif dset == 'val':
            self.names = [i for i in self.names if (int(i.split('_')[0]) >= 50000
                                                    and int(i.split('_')[0]) < 55000)]
        else:
            self.names = [i for i in self.names if int(i.split('_')[0]) >= 55000]
        self.max_src = n_src
        self.n_src = n_src
        self.bg_noise = bg_noise
        
    def shuffle(self):
        random.shuffle(self.names)

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        self.n_src = np.random.randint(2, self.max_src + 1)
        name = name.split('.')[0] + '.pt'
        y = torch.load(os.path.join(self.data_dir + '/pts1', name))
        sources = (y.T)[:int(self.order +  1) ** 2].unsqueeze(0)
        for s in range(1, self.max_src):
            y = torch.load(os.path.join(self.data_dir + '/pts' + str(s + 1), name))
            sources = torch.concatenate([sources, 
                                (y.T)[:int(self.order + 1) ** 2].unsqueeze(0)], axis = 0)
        mixture = torch.mean(sources[:self.n_src], axis=0)  #[(M+1)^2, len]
        mix = mixture
        if self.bg_noise:
            set_snr = np.random.uniform(20, 30) #=10*log10(sum(mix**2) / sum(bg**2))
            set_scale = torch.mean(mix ** 2) / (10 ** (set_snr / 10))  # = sum(bg**2)
            bgnoise = torch.randn(mix.shape).to(mix.device)
            sum_bg = torch.mean(bgnoise ** 2)
            bgnoise = bgnoise / torch.sqrt(sum_bg) * torch.sqrt(set_scale)
            mix = mix + bgnoise
        
        tgt = get_env_points(sources[:, 0:1, :]).squeeze(1) #[ns, T]
        
        if self.n_src < self.max_src:
            tgt[self.n_src:] = 0
        return mix, tgt

class InitLocDataset(data.Dataset):
    def __init__(self, input_dir, dset, n_src=2, order = 1, hoa=True, if_rev = True,
                 sr = 16000, bg_noise = False, **kwargs):
        self.hoa = hoa
        self.rev = if_rev
        self.data_dir = input_dir
        self.names = os.listdir(self.data_dir + '/pts1')
        self.names = [i for i in self.names if int(i.split('_')[0]) < 60000]
        self.len = len(self.names)
        self.order = order
        print(self.len)
        self.dset = dset
        if dset == 'train':
            self.names = [i for i in self.names if int(i.split('_')[0]) < 50000]
        elif dset == 'val':
            self.names = [i for i in self.names if (int(i.split('_')[0]) >= 50000
                                                    and int(i.split('_')[0]) < 55000)]
        else:
            self.names = [i for i in self.names if int(i.split('_')[0]) >= 55000]
        self.max_src = n_src
        self.n_src = n_src
        self.bg_noise = bg_noise
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        # time1 = time.time()
        name = self.names[idx]
        self.n_src = np.random.randint(2, self.max_src + 1)
        name = name.split('.')[0] + '.pt'
        # name = name.split('.')[0] + '.wav'
        y = torch.load(os.path.join(self.data_dir + '/pts1', name))
        sources = (y.T)[:int(self.order +  1) ** 2].unsqueeze(0)
        for s in range(1, self.max_src):
            y = torch.load(os.path.join(self.data_dir + '/pts' + str(s + 1), name))
            sources = torch.concatenate([sources, 
                                (y.T)[:int(self.order + 1) ** 2].unsqueeze(0)], axis = 0)
        mixture = torch.mean(sources[:self.n_src], axis=0)  #[(M+1)^2, len]
        mix = mixture
        if self.bg_noise:
            set_snr = np.random.uniform(20, 30) #=10*log10(sum(mix**2) / sum(bg**2))
            set_scale = torch.mean(mix ** 2) / (10 ** (set_snr / 10))  # = sum(bg**2)
            bgnoise = torch.randn(mix.shape).to(mix.device)
            sum_bg = torch.mean(bgnoise ** 2)
            bgnoise = bgnoise / torch.sqrt(sum_bg) * torch.sqrt(set_scale)
            mix = mix + bgnoise
        
        tgt = sources[:, 0:1, :] #[ns, 1, len]
        tgt = get_env_points(tgt).squeeze(1) #[ns, T]
        
        # print('4', time.time() - time1)
        # time1 = time.time()
        if self.n_src < self.max_src:
            tgt[self.n_src:] = 0
        traj = torch.load(os.path.join(self.data_dir + '/pttraj1',
                                        name)).unsqueeze(0) #[1, 48000, 3]
        for s in range(1, self.max_src):
            # print(os.path.join(self.data_dir + '/s' + str(s + 1), name))
            y = torch.load(os.path.join(self.data_dir + '/pttraj' + str(s + 1), name))
                # sources.append((y.T)[:int(self.order+1)**2])  #[(M+1)^2, len]
            traj = torch.concatenate([traj, 
                                y.unsqueeze(0)], axis = 0)
        dist = torch.norm(traj, dim = -1, keepdim=True)
        traj = traj / dist
        
        return mix, tgt, traj

class STInforDataset_IV(data.Dataset):
    def __init__(self, input_dir, dset, n_src=2, sr = 16000, order = 1, hoa = True, rev = True,
                 multi_ch = False, bg_noise = True, return_ns = False, **kwargs):
        self.data_dir = input_dir
        self.hoa = hoa
        self.rev = rev
        self.names = os.listdir(self.data_dir + '/pts1')
        self.names = [i for i in self.names if int(i.split('_')[0]) < 60000]
        self.len = len(self.names)
        self.order = order
        self.bg_noise = bg_noise
        print(self.len)
        self.dset = dset
        self.return_ns = return_ns
        if dset == 'train':
            self.names = [i for i in self.names if int(i.split('_')[0]) < 50000]
        elif dset == 'val':
            self.names = [i for i in self.names if (int(i.split('_')[0]) >= 50000
                                                    and int(i.split('_')[0]) < 55000)]
        else:
            self.names = [i for i in self.names if int(i.split('_')[0]) >= 55000]
        self.multi_ch = multi_ch
        self.n_src = n_src
        self.max_src = n_src

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        self.n_src = np.random.randint(2, self.max_src + 1)
        y = torch.load(os.path.join(self.data_dir + '/pts1', name))
        sources = (y.T)[:int(self.order +  1) ** 2].unsqueeze(0)
        for s in range(1, self.n_src):
            y = torch.load(os.path.join(self.data_dir + '/pts' + str(s + 1), name))
            sources = torch.concatenate([sources, 
                                (y.T)[:int(self.order + 1) ** 2].unsqueeze(0)], axis = 0)
        mixture = torch.mean(sources[:self.n_src], axis=0)  #[(M+1)^2, len]
        mix = mixture
        if self.bg_noise:
            set_snr = np.random.uniform(20, 30) #=10*log10(sum(mix**2) / sum(bg**2))
            set_scale = torch.mean(mix ** 2) / (10 ** (set_snr / 10))  # = sum(bg**2)
            bgnoise = torch.randn(mix.shape).to(mix.device)
            sum_bg = torch.mean(bgnoise ** 2)
            bgnoise = bgnoise / torch.sqrt(sum_bg) * torch.sqrt(set_scale)
            mix = mix + bgnoise
        
        sel_idx = np.random.randint(0, self.n_src)
        traj = torch.load(os.path.join(self.data_dir + '/pttraj' + str(sel_idx + 1), name))
        sources = sources[sel_idx] #[4, 48000]
        
        traj1 = torch.cat([torch.mean(traj[i:i + 256], dim = 0, keepdim = True) 
                            for i in range(0, traj.shape[0] - 256 + 1, 128)], axis = 0)
        # print(spat_inf.shape)
        traj2 = torch.cat([torch.mean(traj[:128], dim = 0, keepdim = True),
                          traj1, 
                          torch.mean(traj[-128:], dim = 0, keepdim = True)], axis = 0) #[T, 3]
        traj = traj2.permute(1, 0).float()  #[3, T]
        dist = torch.sqrt(traj[0:1] ** 2 + traj[1:2] ** 2 + traj[2:3] ** 2) #[1, T]
        traj = traj / dist
        
        getnorm = sources[0:1] #[1, 48000]
        getnorm = get_env_points(getnorm) #[1, T]
        traj = traj * getnorm  #[3, T]
        
        if not self.multi_ch:
            trg = sources[0:1]
        else:
            trg = sources
        if self.return_ns:
            trg = [trg, self.n_src]
        return [mix, traj], trg

class PrecLocDataset(data.Dataset):
    def __init__(self, input_dir, dset, n_src=2, sr = 16000, order = 1, hoa = True, rev = True,
                 multi_ch = False, bg_noise = True, return_ns = False, **kwargs):
        self.data_dir = input_dir
        self.hoa = hoa
        self.rev = rev
        self.names = os.listdir(self.data_dir + '/pts1')
        self.names = [i for i in self.names if int(i.split('_')[0]) < 60000]
        self.len = len(self.names)
        self.order = order
        self.bg_noise = bg_noise
        print(self.len)
        self.dset = dset
        self.return_ns = return_ns
        if dset == 'train':
            self.names = [i for i in self.names if int(i.split('_')[0]) < 50000]
        elif dset == 'val':
            self.names = [i for i in self.names if (int(i.split('_')[0]) >= 50000
                                                    and int(i.split('_')[0]) < 55000)]
        else:
            self.names = [i for i in self.names if int(i.split('_')[0]) >= 55000]
        self.multi_ch = multi_ch
        self.n_src = n_src
        self.max_src = n_src

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        sources = []
        name = name.split('.')[0] + '.pt'
        # name = name.split('.')[0] + '.wav'
        y = torch.load(os.path.join(self.data_dir + '/pts1', name))
        self.n_src = np.random.randint(2, self.max_src + 1)
        sources = (y.T)[:int(self.order +  1) ** 2].unsqueeze(0)
        for s in range(1, self.n_src):
            y = torch.load(os.path.join(self.data_dir + '/pts' + str(s + 1), name))
            sources = torch.concatenate([sources, 
                            (y.T)[:int(self.order + 1) ** 2].unsqueeze(0)], axis = 0)
        mix = torch.mean(sources, axis=0)  #[(M+1)^2, len]
        if self.bg_noise:
            set_snr = np.random.uniform(20, 30) #=10*log10(sum(mix**2) / sum(bg**2))
            set_scale = torch.mean(mix ** 2) / (10 ** (set_snr / 10))  # = sum(bg**2)
            bgnoise = torch.randn(mix.shape).to(mix.device).float()
            sum_bg = torch.mean(bgnoise ** 2)
            bgnoise = bgnoise / torch.sqrt(sum_bg) * torch.sqrt(set_scale)
            mix = mix + bgnoise
            
        sel_idx = np.random.randint(0, self.n_src)
        traj = torch.load(os.path.join(self.data_dir + '/pttraj' + str(sel_idx + 1),
                                        name)) #[48000, 3]
        
        sources = sources[sel_idx] #[4, 48000]
        
        traj_t = traj #[tt, 3]
        dist = torch.norm(traj_t, dim = -1, keepdim=True)
        traj_t = traj_t / dist
        
        traj1 = torch.cat([torch.mean(traj[i:i + 256], dim = 0, keepdim = True) 
                            for i in range(0, traj.shape[0] - 256 + 1, 128)], axis = 0)
        # print(spat_inf.shape)
        traj2 = torch.cat([torch.mean(traj[:128], dim = 0, keepdim = True),
                          traj1, 
                          torch.mean(traj[-128:], dim = 0, keepdim = True)], axis = 0) #[T, 3]
        traj = traj2.permute(1, 0)  #[3, T]
        dist = torch.norm(traj, dim = 0, keepdim=True)
        traj = traj / dist
        
        getnorm = sources[0:1] #[1, 48000]
        getnorm = get_env_points(getnorm) #[1, T]
        traj_IV = traj * getnorm  #[3, T]
        getnorm_t = get_env_by_inpterp(getnorm.unsqueeze(0)).squeeze(0)
        return mix, traj_IV, sources, traj_t, getnorm_t
    
class MFTSDataset(data.Dataset):
    def __init__(self, input_dir, dset, n_src=2, order = 1, hoa=True, if_rev = True,
                 sr = 16000, bg_noise = True, **kwargs):
        self.hoa = hoa
        self.rev = if_rev
        self.data_dir = input_dir
        self.names = os.listdir(self.data_dir + '/pts1')
        self.names = [i for i in self.names if int(i.split('_')[0]) < 60000]
        self.len = len(self.names)
        self.order = order
        print(self.len)
        self.dset = dset
        if dset == 'train':
            self.names = [i for i in self.names if int(i.split('_')[0]) < 50000]
        elif dset == 'val':
            self.names = [i for i in self.names if (int(i.split('_')[0]) >= 50000
                                                    and int(i.split('_')[0]) < 55000)]
        else:
            self.names = [i for i in self.names if int(i.split('_')[0]) >= 55000]
        self.max_src = n_src
        self.n_src = n_src
        self.bg_noise = bg_noise
        
    def shuffle(self):
        random.shuffle(self.names)

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        self.n_src = np.random.randint(2, self.max_src + 1)
        y = torch.load(os.path.join(self.data_dir + '/pts1', name))
        self.n_src = np.random.randint(2, self.max_src + 1)
        sources = (y.T)[:int(self.order +  1) ** 2].unsqueeze(0)
        for s in range(1, self.max_src):
            y = torch.load(os.path.join(self.data_dir + '/pts' + str(s + 1), name))
            sources = torch.concatenate([sources, 
                            (y.T)[:int(self.order + 1) ** 2].unsqueeze(0)], axis = 0)
        mix = torch.mean(sources[:self.n_src], axis=0)  #[(M+1)^2, len]
        if self.bg_noise:
            set_snr = np.random.uniform(20, 30) #=10*log10(sum(mix**2) / sum(bg**2))
            set_scale = torch.mean(mix ** 2) / (10 ** (set_snr / 10))  # = sum(bg**2)
            bgnoise = torch.randn(mix.shape).to(mix.device).float()
            sum_bg = torch.mean(bgnoise ** 2)
            bgnoise = bgnoise / torch.sqrt(sum_bg) * torch.sqrt(set_scale)
            mix = mix + bgnoise
          
        tgt_wav = sources
        
        tgt_env = torch.from_numpy(sources[:, 0:1, :]).float() #[ns, 1, len]
        tgt_env = get_env_points(tgt_env).squeeze(1) #[ns, T]
        
        if self.n_src < self.max_src:
            tgt_env[self.n_src:] = 0
            
        traj = torch.load(os.path.join(self.data_dir + '/pttraj1',
                                        name)).unsqueeze(0) #[1, 48000, 3]
        for s in range(1, self.max_src):
            # print(os.path.join(self.data_dir + '/s' + str(s + 1), name))
            y = torch.load(os.path.join(self.data_dir + '/pttraj' + str(s + 1), name))
                # sources.append((y.T)[:int(self.order+1)**2])  #[(M+1)^2, len]
            traj = torch.concatenate([traj, 
                                y.unsqueeze(0)], axis = 0)
        
        if self.dset == 'test':
            t60 = torch.load(os.path.join(self.data_dir + '/t60',
                                        name[:-4] + '.pt')) #[4, 48000, 3]
            return mix, tgt_wav, traj, tgt_env, t60
        
        return mix, tgt_wav, traj, tgt_env

class EnvSepDataset(data.Dataset):
    def __init__(self, input_dir, dset, n_src=2, order = 1, hoa=True, if_rev = True,
                 sr = 16000, bg_noise = False, **kwargs):
        self.hoa = hoa
        self.rev = if_rev
        self.data_dir = input_dir
        self.names = os.listdir(self.data_dir + '/pts1')
        self.names = [i for i in self.names if int(i.split('_')[0]) < 60000]
        self.len = len(self.names)
        self.order = order
        print(self.len)
        self.dset = dset
        if dset == 'train':
            self.names = [i for i in self.names if int(i.split('_')[0]) < 50000]
        elif dset == 'val':
            self.names = [i for i in self.names if (int(i.split('_')[0]) >= 50000
                                                    and int(i.split('_')[0]) < 55000)]
        else:
            self.names = [i for i in self.names if int(i.split('_')[0]) >= 55000]
        self.max_src = n_src
        self.n_src = n_src
        self.bg_noise = bg_noise
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        # time1 = time.time()
        name = self.names[idx]
        self.n_src = np.random.randint(2, self.max_src + 1)
        y = torch.load(os.path.join(self.data_dir + '/pts1', name))
        sources = (y.T)[:int(self.order +  1) ** 2].unsqueeze(0)
        for s in range(1, self.max_src):
            y = torch.load(os.path.join(self.data_dir + '/pts' + str(s + 1), name))
            sources = torch.concatenate([sources, 
                            (y.T)[:int(self.order + 1) ** 2].unsqueeze(0)], axis = 0)
        mix = torch.mean(sources[:self.n_src], axis=0)  #[(M+1)^2, len]
        if self.bg_noise:
            set_snr = np.random.uniform(20, 30) #=10*log10(sum(mix**2) / sum(bg**2))
            set_scale = torch.mean(mix ** 2) / (10 ** (set_snr / 10))  # = sum(bg**2)
            bgnoise = torch.randn(mix.shape).to(mix.device).float()
            sum_bg = torch.mean(bgnoise ** 2)
            bgnoise = bgnoise / torch.sqrt(sum_bg) * torch.sqrt(set_scale)
            mix = mix + bgnoise
        # print('2', time.time() - time1)
        # time1 = time.time()
        # print(sources.shape)
        tgt = sources[:, 0:1, :].float() #[ns, 1, len]
        tgt = get_env_points(tgt).squeeze(1) #[ns, T]
        
        # print('4', time.time() - time1)
        # time1 = time.time()
        if self.n_src < self.max_src:
            tgt[self.n_src:] = 0
        return mix, tgt, sources
    
class NoInforDataset(data.Dataset):
    def __init__(self, input_dir, dset, n_src=2, sr = 16000, order = 1, hoa = True, rev = True,
                 multi_ch = True, bg_noise = True, **kwargs):
        self.hoa = hoa
        self.rev = rev
        self.data_dir = input_dir
        self.names = os.listdir(self.data_dir + '/pts1')
        self.names = [i for i in self.names if int(i.split('_')[0]) < 60000]
        self.len = len(self.names)
        self.order = order
        self.multi_ch = multi_ch
        print(self.len)
        self.dset = dset
        if dset == 'train':
            self.names = [i for i in self.names if int(i.split('_')[0]) < 50000]
        elif dset == 'val':
            self.names = [i for i in self.names if (int(i.split('_')[0]) >= 50000
                                                    and int(i.split('_')[0]) < 55000)]
        else:
            self.names = [i for i in self.names if int(i.split('_')[0]) >= 55000]
        self.n_src = n_src
        self.max_src = n_src
        self.bg_noise = bg_noise
        # print(len(self.names))

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        y = torch.load(os.path.join(self.data_dir + '/pts1', name))
        sources = (y.T)[:int(self.order +  1) ** 2].unsqueeze(0)
        for s in range(1, self.n_src):
            y = torch.load(os.path.join(self.data_dir + '/pts' + str(s + 1), name))
            sources = torch.concatenate([sources, 
                            (y.T)[:int(self.order + 1) ** 2].unsqueeze(0)], axis = 0)
        mix = torch.mean(sources[:self.n_src], axis=0)  #[(M+1)^2, len]
        if self.bg_noise:
            set_snr = np.random.uniform(20, 30) #=10*log10(sum(mix**2) / sum(bg**2))
            set_scale = torch.mean(mix ** 2) / (10 ** (set_snr / 10))  # = sum(bg**2)
            bgnoise = torch.randn(mix.shape).to(mix.device).float()
            sum_bg = torch.mean(bgnoise ** 2)
            bgnoise = bgnoise / torch.sqrt(sum_bg) * torch.sqrt(set_scale)
            mix = mix + bgnoise

        if idx % 2 == 0:
            permute = [0, 1]
        else:
            permute = [1, 0]
        trg = sources[permute] #[2, 4, 48000]
        if not self.multi_ch:
            trg = trg[:, 0:1] #[2, 1, 48000]
        return mix, trg

class NoinforLocDataset(data.Dataset):
    def __init__(self, input_dir, dset, n_src=2, order = 1, hoa=True, if_rev = True,
                 sr = 16000, bg_noise = False, **kwargs):
        self.hoa = hoa
        self.rev = if_rev
        self.data_dir = input_dir
        self.names = os.listdir(self.data_dir + '/pts1')
        self.names = [i for i in self.names if int(i.split('_')[0]) < 60000]
        self.len = len(self.names)
        self.order = order
        print(self.len)
        self.dset = dset
        if dset == 'train':
            self.names = [i for i in self.names if int(i.split('_')[0]) < 50000]
        elif dset == 'val':
            self.names = [i for i in self.names if (int(i.split('_')[0]) >= 50000
                                                    and int(i.split('_')[0]) < 55000)]
        else:
            self.names = [i for i in self.names if int(i.split('_')[0]) >= 55000]
        self.max_src = n_src
        self.n_src = n_src
        self.bg_noise = bg_noise
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        # time1 = time.time()
        name = self.names[idx]
        self.n_src = np.random.randint(2, self.max_src + 1)
        # print('0', time.time() - time1)
        # time1 = time.time()
        
        name = name.split('.')[0] + '.pt'
        # name = name.split('.')[0] + '.wav'
        y = torch.load(os.path.join(self.data_dir + '/pts1', name))
        sources = (y.T)[:int(self.order +  1) ** 2].unsqueeze(0)
        for s in range(1, self.n_src):
            # print(os.path.join(self.data_dir + '/s' + str(s + 1), name))
            if self.hoa:
                # y, _ = sf.read(os.path.join(self.data_dir + '/s' + str(s + 1), name))
                y = torch.load(os.path.join(self.data_dir + '/pts' + str(s + 1), name))
                # sources.append((y.T)[:int(self.order+1)**2])  #[(M+1)^2, len]
                sources = torch.concatenate([sources, 
                                (y.T)[:int(self.order + 1) ** 2].unsqueeze(0)], axis = 0)
            else:
                y, _ = sf.read(os.path.join(self.data_dir + '/s0' + str(s + 1), name))
                sources.append(y.T)   #[(M+1)^2, len]
        # print('1', time.time() - time1)
        # time1 = time.time()
        # sources = np.asarray(sources)  #[ns, 4, len]
        # mixture = np.mean(sources, axis=0)  #[(M+1)^2, len]
        mixture = torch.mean(sources, axis=0)  #[(M+1)^2, len]
        mix = mixture
        if self.bg_noise:
            set_snr = np.random.uniform(20, 30) #=10*log10(sum(mix**2) / sum(bg**2))
            set_scale = torch.mean(mix ** 2) / (10 ** (set_snr / 10))  # = sum(bg**2)
            bgnoise = torch.randn(mix.shape).to(mix.device)
            sum_bg = torch.mean(bgnoise ** 2)
            bgnoise = bgnoise / torch.sqrt(sum_bg) * torch.sqrt(set_scale)
            mix = mix + bgnoise
        # print('2', time.time() - time1)
        # time1 = time.time()
        traj = torch.load(os.path.join(self.data_dir + '/pttraj1',
                                        name)).unsqueeze(0) #[1, 48000, 3]
        for s in range(1, self.n_src):
            # print(os.path.join(self.data_dir + '/s' + str(s + 1), name))
            y = torch.load(os.path.join(self.data_dir + '/pttraj' + str(s + 1), name))
                # sources.append((y.T)[:int(self.order+1)**2])  #[(M+1)^2, len]
            traj = torch.concatenate([traj, 
                                y.unsqueeze(0)], axis = 0) #[ns, 48000, 3]
        # dist = torch.sqrt(traj[..., 0:1] ** 2 + traj[..., 1:2] ** 2 + traj[..., 2:3] ** 2) #[1, T]
        dist = torch.norm(traj, dim = -1, keepdim=True) #[ns, 48000, 1]
        traj = traj / dist
        # print('3', time.time() - time1)
        # time1 = time.time()
        return mix, traj