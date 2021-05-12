import numpy as np

import torch


def compute_epe_train(est_flow, batch):
    """
    Compute EPE during training.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    epe : torch.Tensor
        Mean EPE for current batch.

    """

    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    error = error[mask > 0]
    epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
    epe = epe_per_point.mean()

    return epe


def compute_epe(est_flow, batch):
    """
    Compute EPE, accuracy and number of outliers.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    EPE3D : float
        End point error.
    acc3d_strict : float
        Strict accuracy.
    acc3d_relax : float
        Relax accuracy.
    outlier : float
        Percentage of outliers.

    """

    # Extract occlusion mask
    mask = batch["ground_truth"][0].cpu().numpy()[..., 0]

    # Flow
    sf_gt = batch["ground_truth"][1].cpu().numpy()[mask > 0]
    sf_pred = est_flow.cpu().numpy()[mask > 0]

    #
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    EPE3D = l2_norm.mean()

    #
    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)
    acc3d_strict = (
        (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
    )
    acc3d_relax = (
        (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
    )
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier