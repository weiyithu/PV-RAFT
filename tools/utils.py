import os

import torch


def save_checkpoint(model, args, epoch, mode='train'):
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
            'state_dict': model.module.state_dict(),
        }
    else:
        states = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }
    torch.save(states, checkpoint_path)
