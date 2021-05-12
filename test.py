import os
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets.generic import Batch
from datasets.flyingthings3d_hplflownet import FT3D
from datasets.kitti_hplflownet import Kitti
from model.RAFTSceneFlow import RSF
from model.RAFTSceneFlowRefine import RSF_refine
from tools.loss import sequence_loss, compute_loss
from tools.metric import compute_epe


def parse_args():
    parser = argparse.ArgumentParser(description='Testing Argument')
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
    parser.add_argument('--gpus',
                        help='gpus that used for training',
                        default='0',
                        type=str)
    parser.add_argument('--weights',
                        help='checkpoint weights to be loaded',
                        default=None,
                        type=str)
    parser.add_argument('--refine',
                        help='refine mode',
                        action='store_true')
    args = parser.parse_args()

    return args


def testing(args):
    log_dir = os.path.join(args.root, 'experiments', args.exp_path, 'logs')
    log_name = 'TestAlone_' + args.dataset + '.log'
    logging.basicConfig(
        filename=os.path.join(log_dir, log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO
    )
    warnings.filterwarnings('ignore')
    logging.info(args)

    if args.dataset == 'FT3D':
        folder = 'FlyingThings3D_subset_processed_35m'
        dataset_path = os.path.join(args.root, 'data', folder)
        test_dataset = FT3D(root_dir=dataset_path, nb_points=args.max_points, mode='test')
    elif args.dataset == 'KITTI':
        folder = 'kitti_processed'
        dataset_path = os.path.join(args.root, 'data', folder)
        test_dataset = Kitti(root_dir=dataset_path, nb_points=args.max_points)
    else:
        raise NotImplementedError
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=8,
                                 collate_fn=Batch, drop_last=False)

    if not args.refine:
        model = RSF(args).to('cuda')
    else:
        model = RSF_refine(args).to('cuda')
    weight_path = args.weights
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Load checkpoint from {}'.format(weight_path))
        print('Checkpoint epoch {}'.format(checkpoint['epoch']))
        logging.info('Load checkpoint from {}'.format(weight_path))
        logging.info('Checkpoint epoch {}'.format(checkpoint['epoch']))
    else:
        raise RuntimeError(f"=> No checkpoint found at '{weight_path}")

    model.eval()
    loss_test = []
    epe_test = []
    outlier_test = []
    acc3dRelax_test = []
    acc3dStrict_test = []
    test_progress = tqdm(test_dataloader, ncols=150)
    for i, batch_data in enumerate(test_progress):
        batch_data = batch_data.to('cuda')
        with torch.no_grad():
            est_flow = model(batch_data['sequence'], 32)
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

        test_progress.set_description(
            'Testing: Loss: {:.5f} EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f}'.format(
                np.array(loss_test).mean(),
                np.array(epe_test).mean(),
                np.array(outlier_test).mean(),
                np.array(acc3dRelax_test).mean(),
                np.array(acc3dStrict_test).mean()
            )
        )

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
