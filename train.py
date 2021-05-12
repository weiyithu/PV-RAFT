import os
import argparse

from tools.engine import Trainer
from tools.engine_refine import RefineTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Training Argument')
    parser.add_argument('--root',
                        help='workspace path',
                        default='',
                        type=str)
    parser.add_argument('--exp_path',
                        help='specified experiment log path',
                        default=None,
                        type=str)
    parser.add_argument('--dataset',
                        help="choose dataset from 'FT3D' and 'KITTI'",
                        default='FT3D',
                        type=str)
    parser.add_argument('--max_points',
                        help='maximum number of points sampled from a point cloud',
                        default=8192,
                        type=int)
    parser.add_argument('--corr_levels',
                        help='number of correlation pyramid levels',
                        default=3,
                        type=int)
    parser.add_argument('--base_scales',
                        help='voxelize base scale',
                        default=0.25,
                        type=float)
    parser.add_argument('--truncate_k',
                        help='value of truncate_k in corr block',
                        default=512,
                        type=int)
    parser.add_argument('--iters',
                        help='number of iterations in GRU module',
                        default=8,
                        type=int)
    parser.add_argument('--gamma',
                        help='exponential weights',
                        default=0.8,
                        type=float)
    parser.add_argument('--batch_size',
                        help='number of samples in a mini-batch',
                        default=1,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus that used for training',
                        default='0',
                        type=str)
    parser.add_argument('--num_epochs',
                        help='number of epochs for training',
                        default=20,
                        type=int)
    parser.add_argument('--weights',
                        help='checkpoint weights to be loaded',
                        default=None,
                        type=str)
    parser.add_argument('--checkpoint_interval',
                        help='save checkpoint every N epoch',
                        default=5,
                        type=int)
    parser.add_argument('--refine',
                        help='refine mode',
                        action='store_true')
    args = parser.parse_args()

    return args


def main(args):
    print(args)
    if not args.refine:
        trainer = Trainer(args)
    else:
        trainer = RefineTrainer(args)

    for epoch in range(trainer.begin_epoch, args.num_epochs + 1):
        trainer.training(epoch)
        trainer.val_test(epoch, mode='val')
    trainer.val_test(mode='test')


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
