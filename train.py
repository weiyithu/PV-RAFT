import os
import argparse

from tools.engine import Trainer
from tools.engine_refine import RefineTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Training Argument')
    parser.add_argument('--root', default='', type=str)
    parser.add_argument('--exp_path', default=None, type=str)
    parser.add_argument('--dataset', default='FT3D', type=str)
    parser.add_argument('--max_points', default=8192, type=int)
    parser.add_argument('--voxel_size', default=0.05, type=float)

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
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--checkpoint_interval', default=5, type=int)
    parser.add_argument('--refine', action='store_true')

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
