import os
import time
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
import MinkowskiEngine as ME

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.generic import Batch
from datasets.flyingthings3d_hplflownet import FT3D
from datasets.kitti_hplflownet import Kitti
from datasets.dataloader import DistributedInfSampler
from model.RAFTSceneFlow import RSF
from model.RAFTSceneFlowRefine import RSF_refine
from tools.loss import sequence_loss, compute_loss
from tools.metric import compute_epe
from tools.utils import AverageMeter
from train import synchronize


def parse_args():
    parser = argparse.ArgumentParser(description='Testing Argument')
    parser.add_argument('--root', default='', type=str)
    parser.add_argument('--exp_path', default=None, type=str)
    parser.add_argument('--dataset', default='FT3D', type=str)
    parser.add_argument('--max_points', default=8192, type=int)
    parser.add_argument('--voxel_size', default=0.08, type=float)

    parser.add_argument('--corr_levels', default=3, type=int)
    parser.add_argument('--base_scales', default=0.25, type=float)
    parser.add_argument('--truncate_k', default=512, type=int)
    parser.add_argument('--iters', default=8, type=int)
    parser.add_argument('--gamma', default=0.8, type=float)

    parser.add_argument('--bn_momentum', default=0.02, type=float)
    parser.add_argument('--conv1_kernel_size', default=5, type=int)
    parser.add_argument('--nonlinearity', default='ReLU', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--refine', action='store_true')

    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()

    return args


def testing(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 0
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    
    log_dir = os.path.join(args.root, 'experiments', args.exp_path, 'logs')
    log_name = 'TestAlone_' + args.dataset + '.log'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(log_dir, log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO
    )
    warnings.filterwarnings('ignore')
    logging.info(args)

    if not args.refine:
        model = RSF(args)
    else:
        model = RSF_refine(args)
    model = DDP(
        model.cuda(), device_ids=[args.local_rank], output_device=args.local_rank,
        broadcast_buffers=False, find_unused_parameters=True
    )
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    weight_path = args.weights
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
        model.module.load_state_dict(checkpoint['state_dict'])
        if args.local_rank == 0:
            print('Load checkpoint from {}'.format(weight_path))
            print('Checkpoint epoch {}'.format(checkpoint['epoch']))
            logging.info('Load checkpoint from {}'.format(weight_path))
            logging.info('Checkpoint epoch {}'.format(checkpoint['epoch']))
    else:
        raise RuntimeError(f"=> No checkpoint found at '{weight_path}")

    if args.dataset == 'FT3D':
        folder = 'FlyingThings3D_subset_processed_35m'
        dataset_path = os.path.join(args.root, 'data', folder)
        test_dataset = FT3D(root_dir=dataset_path, nb_points=args.max_points, mode='test', voxel_size=args.voxel_size)
    elif args.dataset == 'KITTI':
        folder = 'kitti_processed'
        dataset_path = os.path.join(args.root, 'data', folder)
        test_dataset = Kitti(root_dir=dataset_path, nb_points=args.max_points, voxel_size=args.voxel_size)
    else:
        raise NotImplementedError
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=8,
                                 collate_fn=Batch, drop_last=False,
                                 sampler=DistributedInfSampler(test_dataset, shuffle=False))

    model.eval()
    iter_time = AverageMeter()
    loss_test = []
    epe_test = []
    outlier_test = []
    acc3dRelax_test = []
    acc3dStrict_test = []
    end = time.time()
    max_iter = len(test_dataloader)
    data_iter = test_dataloader.__iter__()
    for i in range(max_iter):
        batch_data = data_iter.next()
        batch_data = batch_data.to('cuda')
        with torch.no_grad():
            est_flow = model(batch_data, 32)
        if not args.refine:
            loss = sequence_loss(est_flow, batch_data)
            epe, acc3d_strict, acc3d_relax, outlier = compute_epe(est_flow[-1], batch_data)
        else:
            loss = compute_loss(est_flow, batch_data)
            epe, acc3d_strict, acc3d_relax, outlier = compute_epe(est_flow, batch_data)

        loss_test.append(loss.cpu())
        epe_test.append(epe)
        outlier_test.append(outlier)
        acc3dRelax_test.append(acc3d_relax)
        acc3dStrict_test.append(acc3d_strict)

        remain_iter = max_iter - i
        iter_time.update(time.time() - end)
        end = time.time()
        running_time = iter_time.sum
        t_mr, t_sr = divmod(running_time, 60)
        t_hr, t_mr = divmod(t_mr, 60)
        running_time = '{:02d}:{:02d}:{:02d}'.format(int(t_hr), int(t_mr), int(t_sr))
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if args.local_rank == 0:
            print('Testing {}/{}: Loss: {:.5f} EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f} Running: {} Remain: {}'.format(
                i, max_iter,
                np.array(loss_test).mean(),
                np.array(epe_test).mean(),
                np.array(outlier_test).mean(),
                np.array(acc3dRelax_test).mean(),
                np.array(acc3dStrict_test).mean(),
                running_time, remain_time
            )
        )

    if args.local_rank == 0:
        print('Test Result: EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f}'.format(
            np.array(epe_test).mean(),
            np.array(outlier_test).mean(),
            np.array(acc3dRelax_test).mean(),
            np.array(acc3dStrict_test).mean()
        ))
        logging.info(
            'Test Result: EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f}'.format(
                np.array(epe_test).mean(),
                np.array(outlier_test).mean(),
                np.array(acc3dRelax_test).mean(),
                np.array(acc3dStrict_test).mean()
            ))


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    testing(args)
