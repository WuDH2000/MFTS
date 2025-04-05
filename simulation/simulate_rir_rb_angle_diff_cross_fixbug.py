import os
import random
# from gpurir.gpuRIR_master import gpuRIR
import gpuRIR
import numpy as np
import soundfile as sf
import acousticTrackingDataset as at_dataset

from tqdm import tqdm
from encode import HOAencode
import matplotlib.pyplot as plt
import torch

print(os.getpid())

def cart2sph(cart):
    xy2 = cart[:,0]**2 + cart[:,1]**2
    sph = np.zeros_like(cart)
    sph[:,0] = np.sqrt(xy2 + cart[:,2]**2)
    sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
    sph[:,2] = np.arctan2(cart[:,1], cart[:,0])
    return sph

class Parameter:
    """ Random parammeter class.
    You can indicate a constant value or a random range in its constructor and then
    get a value acording to that with getValue(). It works with both scalars and vectors.
    """
    def __init__(self, *args):
        if len(args) == 1:
            self.random = False
            self.value = np.array(args[0])
            self.min_value = None
            self.max_value = None
        elif len(args) == 2:
            self. random = True
            self.min_value = np.array(args[0])
            self.max_value = np.array(args[1])
            self.value = None
        else: 
            raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
    
    def getValue(self):
        if self.random:
            return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
        else:
            return self.value

# Room_sz = Parameter([6,6,6], [12,12,12])
# T60 = Parameter(0.2, 0.5)
# Abs_weights = Parameter([0.5]*6, [1.0]*6)
# Array_pos = Parameter([0.2, 0.2, 0.2], [0.8, 0.8, 0.5])
# array_setup = at_dataset.eigenmike_array_setup

Room_sz = Parameter([3, 3, 3], [10, 10, 6])
T60 = Parameter(0.2, 1.0)
Abs_weights = Parameter([0.5]*6, [1.0]*6)
Array_pos = Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.9])
array_setup = at_dataset.eigenmike_array_setup

K = 1024
# windowing = Windowing_multi(K, K*1//2, window=np.hanning)

def get_one_data(source_signals, order = 4, FOA=True, angle_diff=90, static=True):
    # azis = [np.random.rand() * 360 / 180 * np.pi]
    # eles = [90, 90]
    # print(azis)
    # print(eles)
    room_sz = Room_sz.getValue()
    t60 = T60.getValue()
    print(t60)
    rec_t60 = t60
    abs_weights = Abs_weights.getValue()
    beta = gpuRIR.beta_SabineEstimation(room_sz, t60, abs_weights)
    array_pos = Array_pos.getValue() * room_sz
    if array_pos[0] < 1:
        array_pos[0] = 1
    if room_sz[0] - array_pos[0] < 1:
        array_pos[0] = room_sz[0] - 1
    if array_pos[1] < 1:
        array_pos[1] = 1
    if room_sz[1] - array_pos[1] < 1:
        array_pos[1] = room_sz[1] - 1
    if array_pos[2] < 1:
        array_pos[2] = 1
    if room_sz[2] - array_pos[2] < 1:
        array_pos[2] = room_sz[2] - 1
    mic_pos = array_pos + array_setup.mic_pos
    # print(abs_weights)

    src_pos_min = np.array([0.0, 0.0, 0.0])
    src_pos_max = room_sz

    trajectory_num = len(source_signals)
    nb_points = 156
    fs = 16000
    # source_signals = [np.random.rand(3 * fs) for _ in range(trajectory_num)]
    src_pos_inis = [src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min) for _ in range(trajectory_num)]
    # azis = [np.random.rand() * 360 / 180 * np.pi]
    # eles = [(np.random.rand() * 80 + 50) / 180 * np.pi for _ in range(trajectory_num)]
    # # print(azis)
    # # print(eles)
    # # [(np.random.rand() * 80 + 50) / 180 * np.pi for _ in range(2)]
    # t_add_azi = 0
    # for tn in range(trajectory_num - 1):
    #     # print(angle_diff)
    #     add_azi = angle_diff / 180 * np.pi + np.random.rand() * ((180 - angle_diff) / 180 * np.pi)
    #     t_add_azi += add_azi
    #     azis.append(azis[0] + t_add_azi)
    # # print(azis[0] / np.pi * 180, azis[0] / np.pi * 180)
    # # print(eles[0] / np.pi * 180, eles[0] / np.pi * 180)
    # src_pos_inis = [np.array([np.cos(azi) * np.sin(eles[azi_idx]), np.sin(azi) * np.sin(eles[azi_idx]), np.cos(eles[azi_idx])]) + array_pos for azi_idx, azi in enumerate(azis)]
    src_pos_ends = [src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min) for _ in range(trajectory_num)]


    Amaxs = [np.min(np.stack((src_pos_ini - src_pos_min,
            src_pos_max - src_pos_ini,
            src_pos_end - src_pos_min,
            src_pos_max - src_pos_end)),
        axis=0) for src_pos_ini, src_pos_end in zip(src_pos_inis, src_pos_ends)]

    As = [np.random.random(3) * np.minimum(Amax, 1) for Amax in Amaxs]
    ws = [2*np.pi / nb_points * np.random.random(3) * 2 for _ in range(trajectory_num)]
    paired_AW = [(A, w) for A, w in zip(As, ws)]

    traj_pts = [np.array([np.linspace(i, j, nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose() for src_pos_ini, src_pos_end in zip(src_pos_inis, src_pos_ends)]
    traj_pts = [A[0] * np.sin(A[1] * np.arange(nb_points)[:, np.newaxis]) + traj_pt for A, traj_pt in zip(paired_AW, traj_pts)]

    sti_traj_pts = [np.ones((nb_points,1)) * src_pos_ini for src_pos_ini in src_pos_inis]
    static_ratio = 6 if static is True else -1
    traj_pts = [traj_pt if np.random.random(1) >  0.25 * static_ratio else sti_traj_pt for traj_pt, sti_traj_pt in zip(traj_pts, sti_traj_pts)]

    timestamps = np.arange(nb_points) * 3 / nb_points
    t = np.arange(3 * fs) / fs

    trajectory = [np.array([np.interp(t, timestamps, sub_traj_pts[:, i]) for i in range(3)]).transpose() for sub_traj_pts in traj_pts]
    # DOA = [cart2sph(traj - array_pos) [:,1:3] for traj in trajectory]
    traj_xyz = [traj - array_pos for traj in trajectory]

    if t60 == 0:
        Tdiff = 0.1
        Tmax = 0.1
        nb_img = [1, 1, 1]
    else:
        Tdiff = gpuRIR.att2t_SabineEstimator(12, t60) # Use ISM until the RIRs decay 12dB
        Tmax = gpuRIR.att2t_SabineEstimator(40, t60)  # Use diffuse model until the RIRs decay 40dB
        if t60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
        nb_img = gpuRIR.t2n(Tdiff, room_sz)
        #rir最大长度

    nb_mics  = len(mic_pos)
    nb_traj_pts = len(traj_pts[0])
    nb_gpu_calls = min(int(np.ceil(fs * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 1e9 )), nb_traj_pts)
    traj_pts_batch = np.ceil( nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls + 1) ).astype(int)
    multi_s_mic_signals = []
    multi_s0_mic_signals = []
    mix = 0
    scales = []
    for j in range(trajectory_num):
        RIRs_list = [ gpuRIR.simulateRIR(room_sz, beta,
        traj_pts[j][traj_pts_batch[0]:traj_pts_batch[1],:], mic_pos,
                         nb_img, Tmax, fs, Tdiff=Tdiff,
                         orV_rcv=array_setup.mic_orV, mic_pattern=array_setup.mic_pattern) ]
        for i in range(1, nb_gpu_calls):
            RIRs_list += [    gpuRIR.simulateRIR(room_sz, beta, 
                             traj_pts[j][traj_pts_batch[i]:traj_pts_batch[i+1],:], mic_pos,
                             nb_img, Tmax, fs, Tdiff=Tdiff,
                             orV_rcv=array_setup.mic_orV, mic_pattern=array_setup.mic_pattern) ]
        RIRs = np.concatenate(RIRs_list, axis=0)
        s_mic_signals = gpuRIR.simulateTrajectory(source_signals[j], RIRs, timestamps=timestamps, fs=fs)
        s_mic_signals = s_mic_signals[0:len(t), :]
        s_mic_signals0 = s_mic_signals
        if FOA:
            s_mic_signals, _ = HOAencode(s_mic_signals0, 16000, order = order, channel_index=list(range(32)))
        get_scale = (np.max(np.abs(s_mic_signals)) + 1e-4)
        scales.append(get_scale)
        s_mic_signals = s_mic_signals / get_scale#, axis = 0, keepdims = True)
        s_mic_signals0 = s_mic_signals0 / get_scale#, axis = 0, keepdims = True)
        multi_s_mic_signals.append(s_mic_signals)
        multi_s0_mic_signals.append(s_mic_signals0)
        # mix += s_mic_signals
        # print(mix.shape)
    
    if t60 == 0:
        Tdiff = 0.1
        Tmax = 0.1
        nb_img = [1, 1, 1]
    else:
        Tdiff = gpuRIR.att2t_SabineEstimator(12, t60) # Use ISM until the RIRs decay 12dB
        Tmax = gpuRIR.att2t_SabineEstimator(40, t60)  # Use diffuse model until the RIRs decay 40dB
        if t60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
        nb_img = gpuRIR.t2n(Tdiff, room_sz )
        #rir最大长度

    nb_mics  = len(mic_pos)
    nb_traj_pts = len(traj_pts[0])
    nb_gpu_calls = min(int(np.ceil(fs * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 1e9 )), nb_traj_pts)
    traj_pts_batch = np.ceil( nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls + 1) ).astype(int)
    early_multi_s_mic_signals = []
    early_multi_s0_mic_signals = []
    for j in range(trajectory_num):
        RIRs_list = [ gpuRIR.simulateRIR(room_sz, beta,
        traj_pts[j][traj_pts_batch[0]:traj_pts_batch[1],:], mic_pos,
                         nb_img, Tmax, fs, Tdiff=Tdiff,
                         orV_rcv=array_setup.mic_orV, mic_pattern=array_setup.mic_pattern) ]
        for i in range(1, nb_gpu_calls):
            RIRs_list += [    gpuRIR.simulateRIR(room_sz, beta, 
                             traj_pts[j][traj_pts_batch[i]:traj_pts_batch[i+1],:], mic_pos,
                             nb_img, Tmax, fs, Tdiff=Tdiff,
                             orV_rcv=array_setup.mic_orV, mic_pattern=array_setup.mic_pattern) ]
        RIRs = np.concatenate(RIRs_list, axis=0)
        # print(RIRs.shape)
        # plt.figure()
        # plt.plot(RIRs[0, 0])
        # plt.savefig('a_check_rir.png')
        delay = np.argmax(np.abs(RIRs), axis = -1, keepdims = True) #[S, M, 1]
        mask1 = np.ones_like(RIRs) * np.arange(RIRs.shape[-1]) #[S, M, T]
        delay_end = delay + 16000 * 0.05
        mask1 = (mask1 < delay_end).astype(int).astype(np.float32)
        mask2 = np.ones_like(RIRs) * np.arange(RIRs.shape[-1]) #[S, M, T]
        delay_st = delay - 16000 * 0.05
        mask2 = (mask2 > delay_st).astype(int).astype(np.float32)
        # print(RIRs)
        RIRs = RIRs * (mask1 * mask2)
        s_mic_signals = gpuRIR.simulateTrajectory(source_signals[j], RIRs, timestamps=timestamps, fs=fs)
        s_mic_signals = s_mic_signals[0:len(t), :]
        s_mic_signals0 = s_mic_signals
        if FOA:
            s_mic_signals, _ = HOAencode(s_mic_signals0, 16000, order = order, channel_index=list(range(32)))
        get_scale = scales[j]
        s_mic_signals = s_mic_signals / get_scale#, axis = 0, keepdims = True)
        s_mic_signals0 = s_mic_signals0 / get_scale#, axis = 0, keepdims = True)
        early_multi_s_mic_signals.append(s_mic_signals)
        early_multi_s0_mic_signals.append(s_mic_signals0)

    t60 = 0
    if t60 == 0:
        Tdiff = 0.1
        Tmax = 0.1
        nb_img = [1, 1, 1]
    else:
        Tdiff = gpuRIR.att2t_SabineEstimator(12, t60) # Use ISM until the RIRs decay 12dB
        Tmax = gpuRIR.att2t_SabineEstimator(40, t60)  # Use diffuse model until the RIRs decay 40dB
        if t60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
        nb_img = gpuRIR.t2n(Tdiff, room_sz )
    nb_mics  = len(mic_pos)
    nb_traj_pts = len(traj_pts[0])
    nb_gpu_calls = min(int(np.ceil(fs * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 1e9 )), nb_traj_pts)
    traj_pts_batch = np.ceil( nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls + 1) ).astype(int)
    an_multi_s_mic_signals = []
    an_multi_s0_mic_signals = []
    mix = 0
    for j in range(trajectory_num):
        RIRs_list = [ gpuRIR.simulateRIR(room_sz, beta,
        traj_pts[j][traj_pts_batch[0]:traj_pts_batch[1],:], mic_pos,
                         nb_img, Tmax, fs, Tdiff=Tdiff,
                         orV_rcv=array_setup.mic_orV, mic_pattern=array_setup.mic_pattern) ]
        for i in range(1, nb_gpu_calls):
            RIRs_list += [    gpuRIR.simulateRIR(room_sz, beta,
                             traj_pts[j][traj_pts_batch[i]:traj_pts_batch[i+1],:], mic_pos,
                             nb_img, Tmax, fs, Tdiff=Tdiff,
                             orV_rcv=array_setup.mic_orV, mic_pattern=array_setup.mic_pattern) ]
        RIRs = np.concatenate(RIRs_list, axis=0)
        # plt.figure()
        # # print(RIRs.shape)
        # plt.plot(RIRs[0, 0])
        # plt.savefig('a_test_RIR.png')
        an_s_mic_signals = gpuRIR.simulateTrajectory(source_signals[j], RIRs, timestamps=timestamps, fs=fs)
        an_s_mic_signals = an_s_mic_signals[0:len(t), :]
        an_s_mic_signals0 = an_s_mic_signals
        if FOA:
            an_s_mic_signals, _ = HOAencode(an_s_mic_signals0, 16000, order = order, channel_index=list(range(32)))
        get_scale = scales[j]
        an_s_mic_signals = an_s_mic_signals / get_scale#, axis = 0, keepdims = True)
        an_s_mic_signals0 = an_s_mic_signals0 / get_scale#, axis = 0, keepdims = True)
        an_multi_s_mic_signals.append(an_s_mic_signals)
        an_multi_s0_mic_signals.append(an_s_mic_signals0)
    return (True, multi_s_mic_signals, multi_s0_mic_signals,
            early_multi_s_mic_signals, early_multi_s0_mic_signals, 
            an_multi_s_mic_signals, an_multi_s0_mic_signals,
            traj_xyz, room_sz, rec_t60, abs_weights, array_pos)

angle_diff = 30
save_dir = '/data/wdh/ssdata/250224_four_speakers_50ms_scalebymix'
# save_dir='tmp'
# data_dir = './ssdata/fsd18_data_test/'
print(save_dir)
data_dir = '/data/wdh/ssdata/fsd18_data'
sub_labels = os.listdir(data_dir)
if '.DS_Store' in sub_labels:
    sub_labels.remove('.DS_Store')
sub_paths = {
    sub_label: os.listdir(os.path.join(data_dir, sub_label)) for sub_label in sub_labels
}

for sub_label in sub_paths.keys():
    tmp_list = []
    for x in sub_paths[sub_label]:
        if x[-4:] == '.wav':
            tmp_list.append(x)
    sub_paths[sub_label] = tmp_list

st = 47554
ed = 60000
mix_num = 4
sub_files = []
for i in range(mix_num):
    # os.makedirs(os.path.join(save_dir, 's' + str(i + 1)), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'an_s' + str(i + 1)), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'ea_s' + str(i + 1)), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pts' + str(i + 1)), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ptan_s' + str(i + 1)), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ptea_s' + str(i + 1)), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pttraj' + str(i + 1)), exist_ok=True)
os.makedirs(os.path.join(save_dir, 't60'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'abs_weights'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'room_size'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'array_pos'), exist_ok=True)
for idx in tqdm(range(st, ed)):
    flag = False
    while not flag:
        name = str(idx) + '_'
        select_labels = random.sample(sub_labels, mix_num)
        select_idxs = [os.path.join(os.path.join(data_dir, select_label), random.choice(sub_paths[select_label])) for select_label in select_labels]
        name += '_'.join(select_labels)
        source_signals = []
        # sub_files.append(select_idxs)
        for wav_path in select_idxs:
            y, sr = sf.read(wav_path)
            ll = int(3 * sr)  #选择3s
            while len(y) < ll:
                y = np.concatenate([y, y])
            st = random.randint(0, len(y) - ll + 1)
            y = y[st:st + ll]
            source_signals.append(y / (np.std(y) + 1e-10))
        (flag, multi_s_mic_signals, multi_s0_mic_signals, 
        early_multi_s_mic_signals, early_multi_s0_mic_signals, 
        an_multi_s_mic_signals, an_multi_s0_mic_signals, traj_xyz, rs, t60, abs_weights, array_pos) = \
                get_one_data(source_signals, order = 1, FOA = True, 
                             angle_diff=angle_diff, static = False)
    
    print(traj_xyz[0].shape)
    print(traj_xyz[1].shape)

    print(t60)
    t60 = torch.from_numpy(np.array(t60).astype(np.float16))
    torch.save(t60, os.path.join(os.path.join(save_dir, 't60'), name + '.pt'))
    print(os.path.join(os.path.join(save_dir, 't60'), name + '.pt'))
    
    print(rs.shape)
    rs = torch.from_numpy(rs.astype(np.float16))
    torch.save(rs, os.path.join(os.path.join(save_dir, 'room_size'), name + '.pt'))
    print(os.path.join(os.path.join(save_dir, 'room_size'), name + '.pt'))
    
    print(abs_weights.shape)
    abs_weights = torch.from_numpy(abs_weights.astype(np.float16))
    torch.save(abs_weights, os.path.join(os.path.join(save_dir, 'abs_weights'), name + '.pt'))
    print(os.path.join(os.path.join(save_dir, 'abs_weights'), name + '.pt'))
    
    print(array_pos.shape)
    array_pos = torch.from_numpy(array_pos.astype(np.float16))
    torch.save(array_pos, os.path.join(os.path.join(save_dir, 'array_pos'), name + '.pt'))
    print(os.path.join(os.path.join(save_dir, 'array_pos'), name + '.pt'))
    for ii, s in enumerate(multi_s_mic_signals):
        print(s.shape)
        torch.save(torch.from_numpy(s.astype(np.float16)), 
                   os.path.join(os.path.join(save_dir, 'pts' + str(ii + 1)), name + '.pt'))
        print(os.path.join(os.path.join(save_dir, 'pts' + str(ii + 1)), name + '.pt'))
        
        print(early_multi_s_mic_signals[ii].shape)
        torch.save(torch.from_numpy(early_multi_s_mic_signals[ii].astype(np.float16)),
                   os.path.join(os.path.join(save_dir, 'ptea_s' + str(ii + 1)), name + '.pt'))
        print(os.path.join(os.path.join(save_dir, 'ptea_s' + str(ii + 1)), name + '.pt'))
        
        print(an_multi_s_mic_signals[ii].shape)
        torch.save(torch.from_numpy(an_multi_s_mic_signals[ii].astype(np.float16)),
                   os.path.join(os.path.join(save_dir, 'ptan_s' + str(ii + 1)), name + '.pt'))
        print(os.path.join(os.path.join(save_dir, 'ptan_s' + str(ii + 1)), name + '.pt'))
        
        print(traj_xyz[ii].shape)
        torch.save(torch.from_numpy(traj_xyz[ii].astype(np.float16)),
                   os.path.join(os.path.join(save_dir, 'pttraj' + str(ii + 1)), name + '.pt'))
        print(os.path.join(os.path.join(save_dir, 'pttraj' + str(ii + 1)), name + '.pt'))
        