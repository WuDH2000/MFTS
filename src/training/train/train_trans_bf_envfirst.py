"""
The main training script for training on synthetic data
"""

import argparse
import multiprocessing
import os
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # pylint: disable=unused-import
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

from src.helpers import util
from src.training.stinfor_dataset import MFTSDataset as Dataset
from src.training.env_utils import get_env_points, get_env_by_inpterp

print(os.getpid())

iters = 2
sel_index = 0

def nmse(pred, tgt):
    #[B, s, 48000]
    #[s, 48000]
    return 10 * torch.log10(torch.sum((tgt - pred) ** 2, axis = -1) / torch.sum(tgt ** 2, axis = -1))

def ener_weighted_locloss(pred, tgt, est_env):
    getloss = torch.mean((pred * est_env - tgt * est_env) ** 2, axis = (-1, -2))
    return getloss

def test_epoch(model_rs: nn.Module, model_ml:nn.Module, 
                model_sts: nn.Module, model_wl:nn.Module,
                model_bf: nn.Module,
                device: torch.device,
                optimizer: optim.Optimizer,
                test_loader: torch.utils.data.dataloader.DataLoader
                ) -> float:
    """
    Evaluate the network.
    """
    model_sts.eval()
    model_wl.eval()
    model_rs.eval()
    model_ml.eval()
    model_bf.eval()

    # Training loop
    losses = []
    metrics = {}

    with torch.no_grad():
        for batch_idx, (mixed, clean_wav, traj_t, tgt_env_points) in \
                    enumerate(tqdm(test_loader, desc='Test', ncols=100)):
            mixed = mixed.to(device) #[B, 4, 48000]
            clean_wav = clean_wav.to(device)  #[B, ns, 4, 48000]
            traj_t = traj_t.to(device) #[B, ns, 48000, 3]  # to calculate loss
            tgt_env_points = tgt_env_points.to(device) #[B, maxns, T]
        
            output = model_rs(mixed.detach())  #[B, ns, T]
            _, est_env_points = network_rs.loss(output, tgt_env_points, return_est=True)
            est_env_points_s = []
            tgt_env_points_s = []
            label_traj_s = []
            mixed_s = []
            clean_wav_s = []
            
            # print('0', time.time() - time1)
            # time1 = time.time()
            for bb in range(mixed.shape[0]):
                # print(num_directions[bb])
                est_env_points_s_b = est_env_points[bb] #[maxns, T]
                tgt_env_points_s_b = tgt_env_points[bb] #[maxns, T]
                label_traj_s_b = traj_t[bb] #[maxns, len, 3]
                clean_wav_s_b = clean_wav[bb] #[maxns, 4, 48000]
                # print(label_traj_s_b.shape)
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
                
            est_env_points_s = torch.cat(est_env_points_s, axis = 0) #[B, 1, T]
            tgt_env_points_s = torch.cat(tgt_env_points_s, axis = 0)
            label_traj_s = torch.cat(label_traj_s, axis = 0)  #[B, len, 3]
            mixed_s = torch.cat(mixed_s, axis = 0)
            clean_wav_s = torch.cat(clean_wav_s, axis = 0)  #[B, 4, 48000]
                
            est_env_s = get_env_by_inpterp(est_env_points_s) #[B, 1, len]
            tgt_env_s = get_env_by_inpterp(tgt_env_points_s)
                
            loc_in = torch.cat([mixed_s, est_env_s], axis = 1) #
            
            est_traj = model_ml(loc_in.detach()) #[B, T, 3]
                
            est_traj_inf = est_traj #[B, 48000, 3]
            sch_clean_wav = clean_wav_s[:, 0:1] #[B, 1, 48000]
            for iter_ in range(iters):
                
                traj1 = torch.cat([torch.mean(est_traj_inf[:, i:i + 256], dim = 1, keepdim = True) 
                        for i in range(0, est_traj_inf.shape[1] - 256 + 1, 128)], axis = 1)
                # print(spat_inf.shape)
                traj_tf = torch.cat([torch.mean(est_traj_inf[:, :128], dim = 1, keepdim = True),
                                traj1, 
                                torch.mean(est_traj_inf[:, -128:], dim = 1, keepdim = True)], axis = 1) #[B, T, 3]
                traj_tf = traj_tf.permute(0, 2, 1)  #[B, 3, T]
                traj_tf = traj_tf * est_env_points_s  #[B, 3, T]
                # Run through the model
                est_clean_wav = model_sts([mixed_s, traj_tf.detach()]) #[B, 4, 48000]
                output_wl = est_clean_wav / torch.max(torch.max(torch.abs(est_clean_wav),
                        axis = -1)[0], axis = -1)[0].unsqueeze(-1).unsqueeze(-1)
                input_loc = torch.cat([output_wl, mixed_s, est_env_s], axis = 1)
                est_traj = model_wl(input_loc.detach()) #[B, t, 3]
                est_traj_inf = est_traj
                    
                if iter_ == iters - 1:
                    traj1 = torch.cat([torch.mean(est_traj_inf[:, i:i + 256], dim = 1, keepdim = True) 
                            for i in range(0, est_traj_inf.shape[1] - 256 + 1, 128)], axis = 1)
                    # print(spat_inf.shape)
                    traj_tf = torch.cat([torch.mean(est_traj_inf[:, :128], dim = 1, keepdim = True),
                                    traj1, 
                                    torch.mean(est_traj_inf[:, -128:], dim = 1, keepdim = True)], axis = 1) #[B, T, 3]
                    traj_tf = traj_tf.permute(0, 2, 1)  #[B, 3, T]
                    traj_tf = traj_tf * est_env_points_s  #[B, 3, T]
                    bf_wav = torch.cat([est_clean_wav.detach(), mixed_s.detach()], axis = 1) #[B, 8, tt]
                    est_sch_wav = model_bf([bf_wav.detach(), traj_tf.detach()])
                    
                    loss = network_bf.loss(est_sch_wav, sch_clean_wav)

            losses.append(loss.item())

            # Backpropagation
            metrics_batch = network_sts.metrics(est_clean_wav[:, 0:1].detach(),
                                            sch_clean_wav.detach())
            metrics_batch.update(network_wl.metrics(est_traj_inf.detach(), label_traj_s.detach(),
                                                    est_env_s.detach(), tgt_env_s.detach()))
            metrics_batch.update(network_bf.metrics(est_sch_wav.detach(), sch_clean_wav.detach()))
            metrics_batch['loss'] = [loss.item()]
            # print(metrics_batch)
            for k in metrics_batch.keys():
                if not k in metrics:
                    metrics[k] = metrics_batch[k]
                else:
                    metrics[k] += metrics_batch[k]

        avg_metrics = {k: np.mean(metrics[k]) for k in metrics.keys()}
        avg_metrics_str = "Test:"
        for m in avg_metrics.keys():
            avg_metrics_str += ' %s=%.04f' % (m, avg_metrics[m])
        logging.info(avg_metrics_str)

        return avg_metrics

def train_epoch(model_rs: nn.Module, model_ml:nn.Module, 
                model_sts: nn.Module, model_wl:nn.Module,
                model_bf: nn.Module,
                device: torch.device,
                optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader
                ) -> float:
    """
    Train a single epoch.
    """
    # Set the model to training.
    model_sts.eval()
    model_wl.eval()
    model_rs.eval()
    model_ml.eval()
    model_bf.train()

    # Training loop
    losses = []
    metrics = {}

    with tqdm(total=len(train_loader), desc='Train', ncols=100) as t:
        for batch_idx, (mixed, clean_wav, traj_t, tgt_env_points) in enumerate(train_loader):
            mixed = mixed.to(device) #[B, 4, 48000]
            clean_wav = clean_wav.to(device)  #[B, ns, 4, 48000]
            traj_t = traj_t.to(device) #[B, ns, 48000, 3]  # to calculate loss
            tgt_env_points = tgt_env_points.to(device) #[B, maxns, T]
            
            with torch.no_grad():
                output = model_rs(mixed.detach())  #[B, ns, T]
                _, est_env_points = network_rs.loss(output, tgt_env_points, return_est=True)
                est_env_points_s = []
                tgt_env_points_s = []
                label_traj_s = []
                mixed_s = []
                clean_wav_s = []
                
                # print('0', time.time() - time1)
                # time1 = time.time()
                for bb in range(mixed.shape[0]):
                    # print(num_directions[bb])
                    est_env_points_s_b = est_env_points[bb] #[maxns, T]
                    tgt_env_points_s_b = tgt_env_points[bb] #[maxns, T]
                    label_traj_s_b = traj_t[bb] #[maxns, len, 3]
                    clean_wav_s_b = clean_wav[bb] #[maxns, 4, 48000]
                    # print(label_traj_s_b.shape)
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
                    
                est_env_points_s = torch.cat(est_env_points_s, axis = 0) #[B, 1, T]
                tgt_env_points_s = torch.cat(tgt_env_points_s, axis = 0)
                label_traj_s = torch.cat(label_traj_s, axis = 0)  #[B, len, 3]
                mixed_s = torch.cat(mixed_s, axis = 0)
                clean_wav_s = torch.cat(clean_wav_s, axis = 0)  #[B, 4, 48000]
                    
                est_env_s = get_env_by_inpterp(est_env_points_s) #[B, 1, len]
                tgt_env_s = get_env_by_inpterp(tgt_env_points_s)
                    
                loc_in = torch.cat([mixed_s, est_env_s], axis = 1) #
                
                est_traj = model_ml(loc_in.detach()) #[B, T, 3]
                    
                est_traj_inf = est_traj #[B, 48000, 3]
                sch_clean_wav = clean_wav_s[:, 0:1] #[B, 1, 48000]
            for iter_ in range(iters):
                with torch.no_grad():
                    traj1 = torch.cat([torch.mean(est_traj_inf[:, i:i + 256], dim = 1, keepdim = True) 
                            for i in range(0, est_traj_inf.shape[1] - 256 + 1, 128)], axis = 1)
                    # print(spat_inf.shape)
                    traj_tf = torch.cat([torch.mean(est_traj_inf[:, :128], dim = 1, keepdim = True),
                                    traj1, 
                                    torch.mean(est_traj_inf[:, -128:], dim = 1, keepdim = True)], axis = 1) #[B, T, 3]
                    traj_tf = traj_tf.permute(0, 2, 1)  #[B, 3, T]
                    traj_tf = traj_tf * est_env_points_s  #[B, 3, T]
                    # Run through the model
                    est_clean_wav = model_sts([mixed_s, traj_tf.detach()]) #[B, 4, 48000]
                    output_wl = est_clean_wav / torch.max(torch.max(torch.abs(est_clean_wav),
                            axis = -1)[0], axis = -1)[0].unsqueeze(-1).unsqueeze(-1)
                    input_loc = torch.cat([output_wl, mixed_s, est_env_s], axis = 1)
                    est_traj = model_wl(input_loc.detach()) #[B, t, 3]
                    est_traj_inf = est_traj
                    
                if iter_ == iters - 1:
                    traj1 = torch.cat([torch.mean(est_traj_inf[:, i:i + 256], dim = 1, keepdim = True) 
                            for i in range(0, est_traj_inf.shape[1] - 256 + 1, 128)], axis = 1)
                    # print(spat_inf.shape)
                    traj_tf = torch.cat([torch.mean(est_traj_inf[:, :128], dim = 1, keepdim = True),
                                    traj1, 
                                    torch.mean(est_traj_inf[:, -128:], dim = 1, keepdim = True)], axis = 1) #[B, T, 3]
                    traj_tf = traj_tf.permute(0, 2, 1)  #[B, 3, T]
                    traj_tf = traj_tf * est_env_points_s  #[B, 3, T]
                    bf_wav = torch.cat([est_clean_wav.detach(), mixed_s.detach()], axis = 1) #[B, 8, tt]
                    est_sch_wav = model_bf([bf_wav.detach(), traj_tf.detach()])
                    optimizer.zero_grad()
                # Compute loss
                    loss = network_bf.loss(est_sch_wav, sch_clean_wav)
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model_bf.parameters(), 0.5)
                    # Update the weights
                    optimizer.step()

            losses.append(loss.item())

            # Backpropagation
            metrics_batch = network_sts.metrics(est_clean_wav[:, 0:1].detach(),
                                            sch_clean_wav.detach())
            metrics_batch.update(network_wl.metrics(est_traj_inf.detach(), label_traj_s.detach(),
                                                    est_env_s.detach(), tgt_env_s.detach()))
            metrics_batch.update(network_bf.metrics(est_sch_wav.detach(), sch_clean_wav.detach()))
            
            # print(metrics_batch)
            for k in metrics_batch.keys():
                if not k in metrics:
                    metrics[k] = metrics_batch[k]
                else:
                    metrics[k] += metrics_batch[k]
                    
            t.set_postfix(
                        #   loss='%.02f'%loss.item(), 
                          snr = np.mean(np.array(metrics_batch['snr'])),
                          si = np.mean(np.array(metrics_batch['sisnr'])),
                          c_si = np.mean(np.array(metrics_batch['sch_sisnr'])),
                          c_snr = np.mean(np.array(metrics_batch['sch_snr'])),
                          adt = np.mean(np.array(metrics_batch['adt']))
                          )
            t.update()

    avg_metrics = {k: np.mean(metrics[k]) for k in metrics.keys()}
    avg_metrics['loss'] = np.mean(losses)
    avg_metrics_str = "Train:"
    for m in avg_metrics.keys():
        avg_metrics_str += ' %s=%.04f' % (m, avg_metrics[m])
    logging.info(avg_metrics_str)

    return avg_metrics


def train(args: argparse.Namespace):
    """
    Train the network.
    """

    # Load dataset
    data_train = Dataset(**args.train_data)
    # print(args.train_data)
    logging.info("Loaded train dataset at %s containing %d elements" %
                 (args.train_data['input_dir'], len(data_train)))
    data_val = Dataset(**args.val_data)
    logging.info("Loaded val dataset at %s containing %d elements" %
                 (args.val_data['input_dir'], len(data_val)))
    # data_test = Dataset(**args.test_data)
    # logging.info("Loaded test dataset at %s containing %d elements" %
    #              (args.test_data['input_dir'], len(data_test)))

    # Set up the device and workers.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        gpu_ids = args.gpu_ids if args.gpu_ids is not None\
                        else range(torch.cuda.device_count())
        device_ids = [0, 1]#[_ for _ in gpu_ids]
        data_parallel = len(device_ids) > 1
        device = 'cuda:%d' % device_ids[0]
        torch.cuda.set_device(device_ids[0])
        logging.info("Using CUDA devices: %s" % str(device_ids))
    else:
        data_parallel = False
        device = torch.device('cpu')
        logging.info("Using device: CPU")

    # Set multiprocessing params
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if use_cuda else {}

    # Set up data loaders
    #print(args.batch_size, args.eval_batch_size)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=args.eval_batch_size,
                                             shuffle = False,
                                             **kwargs)

    # Set up model
    model_rs = network_rs.Net(dim_output = 2 * 3).to(device)
    model_rs_path = 'experiments/trans_env/mamba_32_bgnoise_fix2src/45.pt'
    # model_rs_path = 'experiments/trans_env/mamba_32_bgnoise_3src/77.pt'
    model_rs.load_state_dict(
        torch.load(model_rs_path, map_location=device)["model_state_dict"]
    )

    ###
    model_ml = network_wl.Net().to(device)
    model_ml_path = 'experiments/trans_initloc/dcsa_2src/86.pt'
    # model_ml_path = 'experiments/trans_initloc/dcsa_3src/51.pt'
    model_ml.load_state_dict(
            torch.load(model_ml_path, map_location=device)["model_state_dict"]
        )

    #########spatinfor sep
    model_sts = network_sts.Net().to(device)
    model_sts_path = "experiments/trans_STIV/mamba_32_bgnoise_fix2src/56.pt"
    # model_sts_path = "experiments/trans_STIV/mamba_32_bgnoise_3src/45.pt"
    model_sts.load_state_dict(
            torch.load(model_sts_path, map_location=device)["model_state_dict"]
    )

    ##########wavinfor loc
    model_wl = network_wl.Net(loc_onsing = False).to(device)
    model_wl_path = 'experiments/trans_precloc/dcsa_precloc_nsrc2/32.pt'
    # model_wl_path = 'experiments/trans_precloc/dcsa_precloc/38.pt'
    model_wl.load_state_dict(
            torch.load(model_wl_path, map_location=device)["model_state_dict"]
        )

    model_bf = network_bf.Net().to(device)

    # Add graph to tensorboard with example train samples
    # _mixed, _label, _ = next(iter(val_loader))
    # args.writer.add_graph(model, (_mixed, _label))

    if use_cuda and data_parallel:
        model_sts = nn.DataParallel(model_sts, device_ids=device_ids)
        model_wl = nn.DataParallel(model_wl, device_ids=device_ids)
        model_ml = nn.DataParallel(model_ml, device_ids=device_ids)
        model_rs = nn.DataParallel(model_rs, device_ids=device_ids)
        model_bf = nn.DataParallel(model_bf, device_ids=device_ids)
        logging.info("Using data parallel model")
    model_sts.to(device)
    model_wl.to(device)
    model_ml.to(device)
    model_rs.to(device)
    model_bf.to(device)

    # Set up the optimizer
    logging.info("Initializing optimizer with %s" % str(args.optim))
    optimizer = network_bf.optimizer(model_bf, **args.optim, data_parallel=data_parallel)
    logging.info('Learning rates initialized to:' + utils.format_lr_info(optimizer))

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **args.lr_sched)
    logging.info("Initialized LR scheduler with params: fix_lr_epochs=%d %s"
                 % (args.fix_lr_epochs, str(args.lr_sched)))

    base_metric = args.base_metric
    train_metrics = {}
    val_metrics = {}

    # Load the model if `args.start_epoch` is greater than 0. This will load the
    # model from epoch = `args.start_epoch - 1`
    assert args.start_epoch >=0, "start_epoch must be greater than 0."
    if args.start_epoch > 0:
        checkpoint_path = os.path.join(args.exp_dir, args.save_model_path, 
                                       '%d.pt' % (args.start_epoch - 1))
        _, train_metrics, val_metrics = utils.load_checkpoint(
            checkpoint_path, model_bf, optim=optimizer, lr_sched=lr_scheduler,
            data_parallel=data_parallel)
        logging.info("Loaded checkpoint from %s" % checkpoint_path)
        logging.info("Learning rates restored to:" + utils.format_lr_info(optimizer))

    try:
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        for epoch in range(args.start_epoch, args.epochs + 1):
            logging.info("Epoch %d:" % epoch)
            os.makedirs(os.path.join(args.exp_dir, args.save_model_path), exist_ok=True)
            checkpoint_file = os.path.join(args.exp_dir, args.save_model_path, '%d.pt' % epoch)
            assert not os.path.exists(checkpoint_file), \
                "Checkpoint file %s already exists" % checkpoint_file
            print("---- begin trianivg")
            curr_train_metrics = train_epoch(model_rs, model_ml, 
                                            model_sts, model_wl,
                                            model_bf,
                                            device,
                                            optimizer,
                                            train_loader)
            #raise KeyboardInterrupt
            curr_test_metrics = test_epoch(model_rs, model_ml, 
                                            model_sts, model_wl,
                                            model_bf,
                                            device,
                                            optimizer,
                                            val_loader)
            # LR scheduler
            if epoch >= args.fix_lr_epochs:
                lr_scheduler.step(curr_test_metrics[base_metric])
                logging.info(
                    "LR after scheduling step: %s" %
                    [_['lr'] for _ in optimizer.param_groups])

            # Write metrics to tensorboard
            args.writer.add_scalars('Train', curr_train_metrics, epoch)
            args.writer.add_scalars('Val', curr_test_metrics, epoch)
            args.writer.flush()

            for k in curr_train_metrics.keys():
                if not k in train_metrics:
                    train_metrics[k] = [curr_train_metrics[k]]
                else:
                    train_metrics[k].append(curr_train_metrics[k])

            for k in curr_test_metrics.keys():
                if not k in val_metrics:
                    val_metrics[k] = [curr_test_metrics[k]]
                else:
                    val_metrics[k].append(curr_test_metrics[k])

            if max(val_metrics[base_metric]) == val_metrics[base_metric][-1]:
                logging.info("Found best validation %s!" % base_metric)
                utils.save_checkpoint(
                    checkpoint_file, epoch, model_bf, optimizer, lr_scheduler,
                    train_metrics, val_metrics, data_parallel)
                logging.info("Saved checkpoint at %s" % checkpoint_file)
            elif epoch % 1 == 0:
                utils.save_checkpoint(
                    checkpoint_file, epoch, model_bf, optimizer, lr_scheduler,
                    train_metrics, val_metrics, data_parallel)
                logging.info("Saved checkpoint at %s" % checkpoint_file)

            # utils.save_graph(train_metrics, val_metrics, args.exp_dir)

        return train_metrics, val_metrics


    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:  # pylint: disable=broad-except
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('exp_dir', type=str,
                        default='./experiments/fsd_mask_label_mult',
                        help="Path to save checkpoints and logs.")

    parser.add_argument('--n_train_items', type=int, default=None,
                        help="Number of items to train on in each epoch")
    parser.add_argument('--n_test_items', type=int, default=None,
                        help="Number of items to test.")
    parser.add_argument('--start_epoch', type=int, default=0,
                        help="Start epoch")
    parser.add_argument('--pretrain_path', type=str,
                        help="Path to pretrained weights")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help="List of GPU ids used for training. "
                        "Eg., --gpu_ids 2 4. All GPUs are used by default.")
    parser.add_argument('--detect_anomaly', dest='detect_anomaly',
                        action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--wandb', dest='wandb', action='store_true',
                        help="Whether to sync tensorboard to wandb")

    args = parser.parse_args()
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    random.seed(230)
    np.random.seed(230)
    if args.use_cuda:
        torch.cuda.manual_seed(230)

    # Set up checkpoints
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    # Load model and training params
    params = utils.Params(os.path.join(args.exp_dir, 'config.json'))
    for k, v in params.__dict__.items():
        vars(args)[k] = v


    utils.set_logger(os.path.join(args.exp_dir, args.save_model_path + '.log'))
    logging.info(str(os.getpid()))

    # Initialize tensorboard writer
    tensorboard_dir = os.path.join(args.exp_dir, 'tensorboard')
    args.writer = SummaryWriter(tensorboard_dir, purge_step=args.start_epoch)
    # if args.wandb:
    #     import wandb
    #     wandb.init(
    #         project='Semaudio', sync_tensorboard=True,
    #         dir=tensorboard_dir, name=os.path.basename(args.exp_dir))

    exec("import %s as network_sts" % args.model_sts)
    exec("import %s as network_wl" % args.model_wl)
    exec("import %s as network_rs" % args.model_rs)
    exec("import %s as network_bf" % args.model_bf)
    logging.info("Imported the model_sts from '%s'." % args.model_sts)
    logging.info("Imported the model_wl from '%s'." % args.model_wl)
    logging.info("Imported the model_rs from '%s'." % args.model_rs)
    logging.info("Imported the model_bf from '%s'." % args.model_bf)
    logging.info(str(args))

    train(args)

    args.writer.close()
    # if args.wandb:
    #     wandb.finish()