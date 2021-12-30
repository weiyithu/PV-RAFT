import os
import time

import torch


def save_checkpoint(model, args, optimizer, epoch, mode='train'):
    if mode == 'train':
        if epoch % args.checkpoint_interval != 0:
            checkpoint_name = 'last_checkpoint.params'
        else:
            checkpoint_name = f'{epoch:03d}.params'
    else:
        checkpoint_name = 'best_checkpoint.params'
    ckpt_dir = os.path.join(args.root, 'experiments', args.exp_path, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    checkpoint_path = os.path.join(ckpt_dir, checkpoint_name)

    if torch.cuda.device_count() > 1:
        states = {
            'epoch': epoch,
            'opt': optimizer.state_dict(),
            'state_dict': model.module.state_dict(),
        }
    else:
        states = {
            'epoch': epoch,
            'opt': optimizer.state_dict(),
            'state_dict': model.state_dict(),
        }
    torch.save(states, checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
