"""
The main training script for training on synthetic data
"""
import sys
sys.path.append("/home/wdh/seld/IEEEStinfor/src/training")

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
from src.helpers import utils
from src.training.stinfor_dataset import InitLocDataset as Dataset
import time

print(os.getpid())

max_filter = torch.nn.MaxPool1d(kernel_size=256, stride=128, padding=128).cuda()
max_filter.requires_grad_ = False
def get_env_points(wav):
    res =  max_filter(torch.abs(wav))
    return res
def get_env_by_inpterp(points, length = 48000):
    sq_flag = False
    if len(points.shape) == 4:
        sq_flag = True
        points = points.squeeze(2)
    res =  torch.nn.functional.interpolate(points, (length), mode = 'linear') #[B, 2, 48000]
    if sq_flag:
        res = res.unsqueeze(2)
    return res
def get_env(wav):
    res = max_filter(torch.abs(wav))
    return torch.nn.functional.interpolate(res, (wav.shape[-1]), mode = 'linear')

def cal_nmse(est_env, tgt_env):
    #[B, 1, T]
    # print(est_env.shape)
    # print(tgt_env.shape)
    res_nmse =  10 * torch.log10(torch.mean((est_env - tgt_env) ** 2, dim = -1) / 
                            (torch.mean(tgt_env ** 2, dim = -1) + 1e-7) + 1e-7)
    # print(res_nmse.shape)
    return res_nmse

def test_epoch(model_rs: nn.Module, device: torch.device,
               test_loader: torch.utils.data.dataloader.DataLoader, 
               model_wl:nn.Module) -> float:
    """
    Evaluate the network.
    """
    torch.cuda.empty_cache()
    model_rs.eval()
    model_wl.eval()
    metrics = {}
 
    with torch.no_grad():
        for batch_idx, (mixed, gt_env_points, gt_traj) in \
                enumerate(tqdm(test_loader, desc='Test', ncols=100)):
            mixed = mixed.to(device).float()
            gt_traj = gt_traj.to(device).float() #[B, ns, T, 3]
            gt_env_points = gt_env_points.to(device).float()
            # print(gt_traj.shape) #[B, ns, len, 3]
            # print(mixed.shape) #[B, 4, 48000]
            # print(gt_env_points.shape) #[B, ns, 376]
            # Run through the model
            output = model_rs(mixed)
            _, est_env_points = network_rs.loss(output, gt_env_points, return_est=True)
            
            est_env_points_s = []
            tgt_env_points_s = []
            label_traj_s = []
            mixed_s = []
            for bb in range(mixed.shape[0]):
                # print(num_directions[bb])
                est_env_points_s_b = est_env_points[bb] #[1, maxns, T]
                tgt_env_points_s_b = gt_env_points[bb] #[1, maxns, T]
                label_traj_s_b = gt_traj[bb] #[1, maxns, len, 3]
                
                exist_tensor = ((torch.max(est_env_points_s_b, axis = -1)[0] > 0.25) 
                                        & (torch.max(tgt_env_points_s_b, axis = -1)[0] > 0.25))
                # print(exist_tensor.shape)
                est_env_points_s_b = est_env_points_s_b[exist_tensor]
                tgt_env_points_s_b = tgt_env_points_s_b[exist_tensor]
                label_traj_s_b = label_traj_s_b[exist_tensor]
                
                num_directions = est_env_points_s_b.shape[0]
                if num_directions == 0:
                    continue
                sel_idx = np.random.randint(0, num_directions)
                
                est_env_points_s_b = est_env_points_s_b[sel_idx:sel_idx + 1] #[1, 1, T]
                tgt_env_points_s_b = tgt_env_points_s_b[sel_idx:sel_idx + 1]
                label_traj_s_b = label_traj_s_b[sel_idx]   #[1, len, 3]
                
                est_env_points_s.append(est_env_points_s_b.unsqueeze(0))
                tgt_env_points_s.append(tgt_env_points_s_b.unsqueeze(0))
                label_traj_s.append(label_traj_s_b.unsqueeze(0))
                mixed_s.append(mixed[bb].unsqueeze(0))
                
                
            est_env_points_s = torch.cat(est_env_points_s, axis = 0) #[B, 1, T]
            tgt_env_points_s = torch.cat(tgt_env_points_s, axis = 0)
            label_traj_s = torch.cat(label_traj_s, axis = 0)
            mixed_s = torch.cat(mixed_s, axis = 0)
            
            est_env_s = get_env_by_inpterp(est_env_points_s) #[B, 1, len]
            tgt_env_s = get_env_by_inpterp(tgt_env_points_s)
            
            
            loc_in = torch.cat([mixed_s, est_env_s], axis = 1) #
                
            est_traj = model_wl(loc_in.detach()) #[B, T, 3]

            loss = network_wl.loss(est_traj, label_traj_s, est_env_s.detach())
            metrics_batch = network_wl.metrics(est_traj.detach(), label_traj_s.detach(),
                                                    est_env_s.detach(), tgt_env_s.detach())
            
            get_nmse = cal_nmse(est_env_points_s.detach(), tgt_env_points_s.detach())
            metrics_batch['l'] = [loss.item()]
            metrics_batch['nmse'] = [np.mean(get_nmse.cpu().detach().numpy())]
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

def train_epoch(model_rs: nn.Module, device: torch.device,
                optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader,
                model_wl:nn.Module) -> float:

    """
    Train a single epoch.
    """
    # Set the model to training.
    torch.cuda.empty_cache()
    model_rs.eval()
    model_wl.train()

    # Training loop
    losses = []
    metrics = {}

    with tqdm(total=len(train_loader), desc='Train', ncols=140) as t:
        for batch_idx, (mixed, gt_env_points, gt_traj) in enumerate(train_loader):
            mixed = mixed.to(device).float()
            # print(mixed.dtype)
            gt_traj = gt_traj.to(device).float() #[B, ns, T, 3]
            gt_env_points = gt_env_points.to(device).float()
            # print('traj', gt_traj.shape) #[B, ns, len, 3]
            # print('mixed', mixed.shape) #[B, 4, 48000]
            # print('env', gt_env_points.shape) #[B, ns, 376]
            
            with torch.no_grad():
                # Run through the model
                # time1 = time.time()
                output = model_rs(mixed.detach())
                # print('*', time.time() - time1)
                # time1 = time.time()
                _, est_env_points = network_rs.loss(output, gt_env_points, return_est=True)
                
                est_env_points_s = []
                tgt_env_points_s = []
                label_traj_s = []
                mixed_s = []
                
                # print('0', time.time() - time1)
                # time1 = time.time()
                for bb in range(mixed.shape[0]):
                    # print(num_directions[bb])
                    est_env_points_s_b = est_env_points[bb] #[maxns, T]
                    tgt_env_points_s_b = gt_env_points[bb] #[maxns, T]
                    label_traj_s_b = gt_traj[bb] #[maxns, len, 3]
                    # print(label_traj_s_b.shape)
                    exist_tensor = ((torch.max(est_env_points_s_b, axis = -1)[0] > 0.25) 
                                            & (torch.max(tgt_env_points_s_b, axis = -1)[0] > 0.25))
                    # print(exist_tensor.shape)
                    est_env_points_s_b = est_env_points_s_b[exist_tensor]
                    tgt_env_points_s_b = tgt_env_points_s_b[exist_tensor]
                    label_traj_s_b = label_traj_s_b[exist_tensor]
                    
                    num_directions = est_env_points_s_b.shape[0]
                    if num_directions == 0:
                        continue
                    sel_idx = np.random.randint(0, num_directions)
                    
                    est_env_points_s_b = est_env_points_s_b[sel_idx:sel_idx + 1] #[1, T]
                    tgt_env_points_s_b = tgt_env_points_s_b[sel_idx:sel_idx + 1]
                    label_traj_s_b = label_traj_s_b[sel_idx]   #[len, 3]
                    
                    est_env_points_s.append(est_env_points_s_b.unsqueeze(0))
                    tgt_env_points_s.append(tgt_env_points_s_b.unsqueeze(0))
                    label_traj_s.append(label_traj_s_b.unsqueeze(0))
                    mixed_s.append(mixed[bb].unsqueeze(0))
                    
                est_env_points_s = torch.cat(est_env_points_s, axis = 0) #[B, 1, T]
                tgt_env_points_s = torch.cat(tgt_env_points_s, axis = 0)
                label_traj_s = torch.cat(label_traj_s, axis = 0)
                mixed_s = torch.cat(mixed_s, axis = 0)
                
                est_env_s = get_env_by_inpterp(est_env_points_s) #[B, 1, len]
                tgt_env_s = get_env_by_inpterp(tgt_env_points_s)
                
                
                loc_in = torch.cat([mixed_s, est_env_s], axis = 1) #
            # print('1', time.time() - time1)
            # time1 = time.time()
            
            est_traj = model_wl(loc_in.detach()) #[B, T, 3]
            # print('2', time.time() - time1)
            # time1 = time.time()
            # Reset grad
            optimizer.zero_grad()
            loss = network_wl.loss(est_traj, label_traj_s, est_env_s.detach())
            losses.append(loss.item())

            # print('3', time.time() - time1)
            # time1 = time.time()
            # Backpropagation
            loss.backward()
            # print('4', time.time() - time1)
            # time1 = time.time()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_wl.parameters(), 0.5)

            # Update the weights
            optimizer.step()

            metrics_batch = network_wl.metrics(est_traj.detach(), label_traj_s.detach(),
                                                    est_env_s.detach(), tgt_env_s.detach())
            
            get_nmse = cal_nmse(est_env_points_s.detach(), tgt_env_points_s.detach())
            metrics_batch['l'] = [loss.item()]
            metrics_batch['nmse'] = [np.mean(get_nmse.cpu().detach().numpy())]
            # print(metrics_batch)
            for k in metrics_batch.keys():
                if not k in metrics:
                    metrics[k] = metrics_batch[k]
                else:
                    metrics[k] += metrics_batch[k]
            print_dict = dict()
            for mn in metrics_batch:
                print_dict[mn] = np.mean(np.array(metrics_batch[mn]))
            # torch.cuda.empty_cache()
            # Show current loss in the progress meter
            # time.sleep(0.5)
            t.set_postfix(print_dict)
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
    print(args.train_data)
    logging.info("Loaded train dataset at %s containing %d elements" %
                 (args.train_data['input_dir'], len(data_train)))
    data_val = Dataset(**args.val_data)
    logging.info("Loaded test dataset at %s containing %d elements" %
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
                                               shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=args.eval_batch_size,
                                             **kwargs)
    # test_loader = torch.utils.data.DataLoader(data_test,
    #                                          batch_size=args.eval_batch_size,
    #                                          **kwargs)
   
    model_rs = network_rs.Net(dim_output = 2 * 2)
    # print(model)
    model_wl = network_wl.Net()
    # model_wl.half()
    print(model_rs)
    
    model_rs_path = 'experiments/trans_env/mamba_32_bgnoise_fix2src/45.pt'
    model_rs.load_state_dict(
        torch.load(model_rs_path, map_location=device)["model_state_dict"]
    )
    # model_rs.half()
    # Add graph to tensorboard with example train samples
    # _mixed, _label, _ = next(iter(val_loader))
    # args.writer.add_graph(model, (_mixed, _label))

    if use_cuda and data_parallel:
        model_rs = nn.DataParallel(model_rs, device_ids=device_ids)
        model_wl = nn.DataParallel(model_wl, device_ids=device_ids)
        logging.info("Using data parallel model")
    model_rs.to(device)
    model_wl.to(device)

    # Set up the optimizer
    logging.info("Initializing optimizer with %s" % str(args.optim))
    optimizer = network_wl.optimizer(model_wl, **args.optim, data_parallel=data_parallel)
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
            checkpoint_path, model_wl, optim=optimizer, lr_sched=lr_scheduler,
            data_parallel=data_parallel)
        logging.info("Loaded checkpoint from %s" % checkpoint_path)
        logging.info("Learning rates restored to:" + utils.format_lr_info(optimizer))

    # Training loop
    try:
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        for epoch in range(args.start_epoch, args.epochs + 1):
            logging.info("Epoch %d:" % epoch)
            os.makedirs(os.path.join(args.exp_dir, args.save_model_path), exist_ok=True)
            checkpoint_file = os.path.join(args.exp_dir, args.save_model_path, '%d.pt' % epoch)
            assert not os.path.exists(checkpoint_file), \
                "Checkpoint file %s already exists" % checkpoint_file
            #print("---- begin trianivg")
            curr_train_metrics = train_epoch(model_rs, device, optimizer,
                                             train_loader, model_wl)
            #raise KeyboardInterrupt
            curr_test_metrics = test_epoch(model_rs, device, val_loader, model_wl)
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

            if min(val_metrics[base_metric]) == val_metrics[base_metric][-1]:
                logging.info("Found best validation %s!" % base_metric)

                utils.save_checkpoint(
                    checkpoint_file, epoch, model_wl, optimizer, lr_scheduler,
                    train_metrics, val_metrics, data_parallel)
                logging.info("Saved checkpoint at %s" % checkpoint_file)
                # curr_test_metrics = test_epoch(model_rs, device, test_loader, model_wl)

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

    exec("import %s as network_rs" % args.model_rs)
    logging.info("Imported the model from '%s'." % args.model_rs)
    exec("import %s as network_wl" % args.model_wl)
    logging.info("Imported the model_wl from '%s'." % args.model_wl)

    train(args)

    args.writer.close()
