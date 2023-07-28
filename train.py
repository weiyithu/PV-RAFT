import os
import argparse
import MinkowskiEngine as ME

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tools.engine import Trainer
from model.dpv_raft import DPV_RAFT


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description='Training Argument')
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
    parser.add_argument('--drop_thresh', default=0.95, type=float)

    parser.add_argument('--bn_momentum', default=0.02, type=float)
    parser.add_argument('--conv1_kernel_size', default=5, type=int)
    parser.add_argument('--nonlinearity', default='ReLU', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--checkpoint_interval', default=5, type=int)

    parser.add_argument('--local_rank', default=0, type=int)

    args = parser.parse_args()

    return args


def main(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 0
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    model = DPV_RAFT(args)
    model = model.cuda()

    ddp_model = DDP(
        model, device_ids=[args.local_rank], output_device=args.local_rank,
        broadcast_buffers=False, find_unused_parameters=True
    )
    ddp_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(ddp_model)

    trainer = Trainer(args, ddp_model)

    for epoch in range(trainer.begin_epoch, args.num_epochs + 1):
        trainer.training(epoch)
        trainer.val_test(epoch, mode='val')
    trainer.val_test(mode='test')


if __name__ == '__main__':
    args = parse_args()
    if args.local_rank == 0:
        print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
