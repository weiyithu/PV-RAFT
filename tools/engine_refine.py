import os
import time
import logging
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.generic import Batch
from datasets.flyingthings3d_hplflownet import FT3D
from datasets.dataloader import DistributedInfSampler
from tools.loss import compute_loss
from tools.metric import compute_epe_train, compute_epe
from tools.utils import save_checkpoint, AverageMeter


class RefineTrainer(object):
    def __init__(self, args, model, mode='Train'):
        self.args = args
        self.root = args.root
        self.exp_path = args.exp_path
        self.dataset = args.dataset
        self.log_dir = None
        self.summary_writer = None

        if args.local_rank == 0:
            self._log_init(mode)

        if self.dataset == 'FT3D':
            folder = 'FlyingThings3D_subset_processed_35m'
            dataset_path = os.path.join(self.root, 'data', folder)
            self.train_dataset = FT3D(root_dir=dataset_path, nb_points=args.max_points, mode='train')
            self.val_dataset = FT3D(root_dir=dataset_path, nb_points=args.max_points, mode='val')
            self.test_dataset = FT3D(root_dir=dataset_path, nb_points=args.max_points, mode='test')
        else:
            raise NotImplementedError

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=int(args.batch_size // len(args.gpus.split(','))),
                                           num_workers=8, collate_fn=Batch, drop_last=True,
                                           sampler=DistributedInfSampler(self.train_dataset, shuffle=True))
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=1, num_workers=8,
                                         collate_fn=Batch, drop_last=False,
                                         sampler=DistributedInfSampler(self.val_dataset, shuffle=False))
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=1, num_workers=8,
                                         collate_fn=Batch, drop_last=False,
                                         sampler=DistributedInfSampler(self.test_dataset, shuffle=False))

        self.model = model

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=args.num_epochs*len(self.train_dataset))

        self.begin_epoch = 1
        self._load_weights()

        if self.begin_epoch > 0:
            for _ in range(self.begin_epoch):
                self.lr_scheduler.step()

        self.best_val_epe = 10

    def _log_init(self, mode='Train'):
        if self.exp_path is None:
            self.exp_path = datetime.now().strftime("exp-%y_%m_%d-%H_%M_%S_%f")
            self.args.exp_path = self.exp_path
        self.exp_path = os.path.join(self.root, 'experiments', self.exp_path)
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)

        log_dir = os.path.join(self.exp_path, 'logs')
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_name = mode + '_' + self.dataset + '.log'
        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=os.path.join(log_dir, log_name),
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO
        )
        warnings.filterwarnings('ignore')

        ckpt_dir = os.path.join(self.exp_path, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        logging.info(self.args)
        logging.info('')

    def _load_weights(self, test_best=False):
        if self.args.weights is not None:
            weight_path = os.path.join(self.root, 'experiments', self.args.weights, 'checkpoints', 'best_checkpoint.params')
            if os.path.exists(weight_path):
                checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
                if torch.cuda.device_count() > 1:
                    self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("Load checkpoint from {}".format(weight_path))
            else:
                raise RuntimeError(f"=> No checkpoint found at '{self.args.weights}")
        if test_best:
            weight_path = os.path.join(self.root, 'experiments', self.args.exp_path, 'checkpoints',
                                       'best_checkpoint.params')
            if os.path.exists(weight_path):
                checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
                if torch.cuda.device_count() > 1:
                    self.model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint['state_dict'])
                print("Load checkpoint from {}".format(weight_path))
            else:
                raise RuntimeError(f"=> No checkpoint found at '{weight_path}")

    def training(self, epoch):
        self.model.train()
        iter_time = AverageMeter()
        if self.summary_writer is None and self.args.local_rank == 0:
            self.summary_writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        loss_train = []
        epe_train = []

        end = time.time()
        max_iter = len(self.train_dataloader)
        data_iter = self.train_dataloader.__iter__()
        for i in range(max_iter):
            batch_data = data_iter.next()
            global_step = epoch * len(self.train_dataloader) + i
            batch_data = batch_data.to('cuda')

            self.optimizer.zero_grad()
            est_flow = self.model(batch_data, num_iters=self.args.iters, drop_thresh=self.args.drop_thresh)
            loss = compute_loss(est_flow, batch_data)
            loss.backward()
            self.optimizer.step()

            epe = compute_epe_train(est_flow, batch_data)
            loss_train.append(loss.detach().cpu())
            epe_train.append(epe.detach().cpu())

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

            if self.args.local_rank == 0:
                self.summary_writer.add_scalar(
                    tag='Train/Loss',
                    scalar_value=np.array(loss_train).mean(),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Train/EPE',
                    scalar_value=np.array(epe_train).mean(),
                    global_step=global_step
                )

            if self.args.local_rank == 0:
                print('Train Epoch {}: Iter: {}/{} Loss: {:.5f} EPE: {:.5f} Running: {} Remain: {}'.format(
                    epoch,
                    i, max_iter,
                    np.array(loss_train).mean(),
                    np.array(epe_train).mean(),
                    running_time, remain_time)
                )

        self.lr_scheduler.step()
        if self.args.local_rank == 0:
            save_checkpoint(self.model, self.args, self.optimizer, self.lr_scheduler, self.best_val_epe, epoch, 'train')
            logging.info('Train Epoch {}: Loss: {:.5f} EPE: {:.5f}'.format(
                        epoch,
                        np.array(loss_train).mean(),
                        np.array(epe_train).mean()
                    ))

    def val_test(self, epoch=0, mode='val'):
        self.model.eval()
        iter_time = AverageMeter()
        loss_run = []
        epe_run = []
        outlier_run = []
        acc3dRelax_run = []
        acc3dStrict_run = []

        if mode == 'val':
            run_dataloader = self.val_dataloader
            run_logstr = 'Val'
        else:
            run_dataloader = self.test_dataloader
            run_logstr = 'Test'
            self._load_weights(test_best=True)
        end = time.time()
        max_iter = len(run_dataloader)
        data_iter = run_dataloader.__iter__()
        for i in range(max_iter):
            batch_data = data_iter.next()
            global_step = epoch * len(run_dataloader) + i
            batch_data = batch_data.to('cuda')

            with torch.no_grad():
                est_flow = self.model(p=batch_data, num_iters=self.args.iters, drop_thresh=self.args.drop_thresh)

            loss = compute_loss(est_flow, batch_data)
            epe, acc3d_strict, acc3d_relax, outlier = compute_epe(est_flow, batch_data)
            loss_run.append(loss.cpu())
            epe_run.append(epe)
            outlier_run.append(outlier)
            acc3dRelax_run.append(acc3d_relax)
            acc3dStrict_run.append(acc3d_strict)

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

            if mode == 'val' and self.args.local_rank == 0:
                self.summary_writer.add_scalar(
                    tag='Val/Loss',
                    scalar_value=np.array(loss_run).mean(),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Val/EPE',
                    scalar_value=np.array(epe_run).mean(),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Val/Outlier',
                    scalar_value=np.array(outlier_run).mean(),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Val/Acc3dRelax',
                    scalar_value=np.array(acc3dRelax_run).mean(),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Val/Acc3dStrict',
                    scalar_value=np.array(acc3dStrict_run).mean(),
                    global_step=global_step
                )

            if self.args.local_rank == 0:
                print(run_logstr + ' Epoch {}: Iter: {}/{} Loss: {:.5f} EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f} Running {} Remain: {}'.format(
                    epoch,
                    i, max_iter,
                    np.array(loss_run).mean(),
                    np.array(epe_run).mean(),
                    np.array(outlier_run).mean(),
                    np.array(acc3dRelax_run).mean(),
                    np.array(acc3dStrict_run).mean(),
                    running_time, remain_time),
                )

        if mode == 'val' and self.args.local_rank == 0:
            if np.array(epe_run).mean() < self.best_val_epe:
                self.best_val_epe = np.array(epe_run).mean()
                save_checkpoint(self.model, self.args, self.optimizer, self.lr_scheduler, self.best_val_epe, epoch, 'val')
            logging.info(
                'Val Epoch {}: Loss: {:.5f} EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f}'.format(
                    epoch,
                    np.array(loss_run).mean(),
                    np.array(epe_run).mean(),
                    np.array(outlier_run).mean(),
                    np.array(acc3dRelax_run).mean(),
                    np.array(acc3dStrict_run).mean()
                ))
            logging.info('Best EPE: {:.5f}'.format(self.best_val_epe))
        if mode == 'test' and self.args.local_rank == 0:
            print('Test Result: EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f}'.format(
                np.array(epe_run).mean(),
                np.array(outlier_run).mean(),
                np.array(acc3dRelax_run).mean(),
                np.array(acc3dStrict_run).mean()
            ))
            logging.info(
                'Test Result: EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f}'.format(
                    np.array(epe_run).mean(),
                    np.array(outlier_run).mean(),
                    np.array(acc3dRelax_run).mean(),
                    np.array(acc3dStrict_run).mean()
            ))
