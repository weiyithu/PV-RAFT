import torch


def sequence_loss(est_flow, batch, gamma=0.8):
    n_predictions = len(est_flow)
    flow_loss = 0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = compute_loss(est_flow[i], batch)
        flow_loss += i_weight * i_loss

    return flow_loss


def compute_loss(est_flow, batch):
    """
    Compute training loss.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    loss : torch.Tensor
        Training loss for current batch.

    """

    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    error = error[mask > 0]
    loss = torch.mean(torch.abs(error))

    return loss
