import os
import trimesh
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
    pc1_set = []
    pc2_set = []
    est_flow_set = []
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
        pc1_set.append(batch_data['sequence'][0])
        pc2_set.append(batch_data['sequence'][1])
        est_flow_set.append(est_flow)

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
    
    # save most error sample
    idx = np.argsort(-np.array(epe_test))[:10]
    vis_dir = os.path.join(args.root, 'experiments', args.exp_path, 'vis', args.dataset)
    os.makedirs(vis_dir, exist_ok=True)
    color_lib = np.array([[0, 0, 255], [255, 0, 0], [0, 255, 0]])
    for i in idx:
        pc1 = pc1_set[i]
        pc2 = pc2_set[i]
        pc_pred = pc1 + est_flow_set[i]
        pc_save = torch.cat([pc1, pc2, pc_pred], dim=0).detach().cpu().numpy()
        color_save = np.stack([np.expand_dims(color_lib[0], 0).repeat(pc1.shape[1], axis=0),
                                np.expand_dims(color_lib[1], 0).repeat(pc1.shape[1], axis=0),
                                np.expand_dims(color_lib[2], 0).repeat(pc1.shape[1], axis=0)])
        save_ply(pc_save.reshape(-1, 3), color_save.reshape(-1, 3), vis_dir, i)
    print(idx)
    logging.info(idx)
    print(np.array(epe_test)[idx])
    logging.info(np.array(epe_test)[idx])

def save_ply(pc, colors, dir, idx):
    path = os.path.join(dir, str(idx) + '.ply')
    pcd = trimesh.points.PointCloud(vertices=pc, colors=colors)
    pcd.export(path)



if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    testing(args)
