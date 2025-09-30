"""Loss functions.

Definition of torch loss functions and metrics for model training and evaluation.
"""
from enum import Enum

import torch
import torchmetrics
from torch import nn


#################################################
#                Classes
#################################################
class AcuracyLoss(nn.Module):
    """Accuracy Loss Function."""

    def __init__(self):
        super().__init__()

    # pred => Predictions (logits, B, 1, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        """Ingeritence forward method."""
        return accuracy_loss(pred, gt)

    # end def


class FScore(nn.Module):
    """FScore Loss Function."""

    def __init__(self):
        super().__init__()

    # pred => Predictions (logits, B, 1, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        """Ongeritence forward method."""
        return f1_metric(pred, gt)

    # end def


class IoULoss(nn.Module):
    """IoULoss Loss Function."""

    def __init__(self, stable=True):
        super().__init__()
        self.stable = stable

    # pred => Predictions (logits, B, 1, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        """Ongeritence forward method."""
        # return 1.0 - IoUMetric(pred, gt, self.softmax)
        # Compute the negative log loss for stable training.
        return stable_iou_loss(pred, gt) if self.stable else iou_loss(pred, gt)

    # end def


class DiceLoss(nn.Module):
    """DiceLoss Loss Function."""

    def __init__(self):
        super().__init__()

    # pred => Predictions (logits, B, 1, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        """Ongeritence forward method."""
        return dice_loss(pred, gt)

    # end def


#################################################
#                 Actual funuctions
#################################################
def accuracy_loss(pred, gt, threshold=0.5):
    """Compute the accuracy loss by comparing the predicted values with the ground truth.

    The accuracy is calculated using a specified threshold to determine the correct classification.

    Args:
        pred (torch.Tensor): The predicted output from the model.
        gt (torch.Tensor): The ground truth values.
        threshold (float, optional): The threshold for classification. Defaults to 0.5.

    Returns:
        float: The mean accuracy loss based on the threshold.
    """
    return (accuracy_metric(pred, gt, threshold)).mean()


def f1_metric(pred, gt):
    """Compute the F1 score metric for binary classification.

    This function calculates the F1 score using the predicted values and the ground truth,
    viewed as flattened tensors for binary classification tasks.

    Args:
        pred (torch.Tensor): The predicted output from the model.
        gt (torch.Tensor): The ground truth values.

    Returns:
        float: The mean F1 score based on the predicted values and ground truth.
    """
    f1 = torchmetrics.F1Score(task="binary", num_classes=1, average="macro")
    return f1(pred.view(-1), gt.view(-1)).mean()


def stable_iou_loss(pred, gt):
    """Compute stable IoU loss.

    Compute the stable Intersection over Union (IoU) loss between
    the predicted and ground truth values.

    This function calculates the IoU, applies a logarithmic transformation, and returns
    the negative mean loss to be minimized during training.

    Args:
        pred (torch.Tensor): The predicted output from the model.
        gt (torch.Tensor): The ground truth values.

    Returns:
        float: The negative mean log of the IoU value, representing the stable IoU loss.
    """
    epsilon = 1e-5
    return -(torch.log(iou_metric(pred, gt) + epsilon)).mean()


def iou_loss(pred, gt):
    """Compute IoU loss.

    Compute the Intersection over Union (IoU) loss between the predicted
    and ground truth values. This function calculates the IoU metric and returns
    the difference from 1, representing the loss to be minimized.

    Args:
        pred (torch.Tensor): The predicted output from the model.
        gt (torch.Tensor): The ground truth values.

    Returns:
        float: The mean IoU loss, calculated as 1 minus the IoU value.
    """
    return (1.0 - iou_metric(pred, gt)).mean()


def dice_loss(pred, gt):
    """Compute the DICE loss between the predicted and ground truth values.

    This function calculates the DICE metric and returns the difference from 1,
    representing the loss to be minimized.

    Args:
        pred (torch.Tensor): The predicted output from the model.
        gt (torch.Tensor): The ground truth values.

    Returns:
        float: The mean DICE loss, calculated as 1 minus the DICE value.
    """
    return (1.0 - dice_metric(pred, gt)).mean()


def accuracy_metric(pred, gt, threshold=0.5):
    """Compute the accuracy metric between the predicted and ground truth values.

    This function calculates the accuracy by comparing the predicted and ground truth values
    using a specified threshold for binary classification.

    Args:
        pred (torch.Tensor): The predicted output from the model.
        gt (torch.Tensor): The ground truth values.
        threshold (float, optional): The threshold for binary classification. Defaults to 0.5.

    Returns:
        float: The accuracy metric, representing the proportion of correct predictions.
    """
    pred_bin = (pred > threshold).float()
    correct_pixels = (pred_bin == gt).sum()
    total_pixels = torch.numel(gt)
    return correct_pixels / total_pixels


def iou_metric(pred, gt):
    """Compute the IoU metric.

    Compute the Intersection over Union (IoU) metric between the predicted and ground truth values.

    This function calculates the IoU by computing the intersection and
    union between the predicted and ground truth values, and returns the IoU value.

    Args:
    pred (torch.Tensor): The predicted output from the model.
    gt (torch.Tensor): The ground truth values.

    Returns:
        torch.Tensor: The IoU value for each batch element.
    """
    intersection = gt * pred
    union = (
        gt + pred - intersection
    )  # bc you sum 1+1 in the intersection pixels (both are 1) so substract

    # Compute the sum over all the dimensions except for the batch dimension.
    # add epsilon to avoid division by 0
    epsilon = 1e-5
    iou = (intersection.sum(dim=(2, 3)) + epsilon) / (union.sum(dim=(2, 3)) + epsilon)

    return iou


def dice_metric(pred, gt):
    """Compute the DICE coefficient between the predicted and ground truth values.

    This function calculates the DICE score by evaluating the intersection
    between the predicted and ground truth values.

    Args:
    pred (torch.Tensor): The predicted output from the model.
    gt (torch.Tensor): The ground truth values.

    Returns:
        torch.Tensor: The DICE coefficient for each batch element.
    """
    intersection = gt * pred
    # Compute the sum over all the dimensions except for the batch dimension.
    # add epsilon to avoid division by 0
    epsilon = 1e-5
    dice = (
        (intersection.sum(dim=(2, 3)) + epsilon)
        * 2
        / (gt.sum(dim=(2, 3)) + pred.sum(dim=(2, 3)) + epsilon)
    )

    return dice


#################################################
#                 Enum to choose
#################################################


class Loss(Enum):
    """Loss enum."""

    ACURACY = AcuracyLoss()
    FSCORE = FScore()
    CROSS_ENTROPY = nn.BCEWithLogitsLoss()
    IOU_NORMAL = IoULoss(stable=False)
    IOU_STABLE = IoULoss(stable=True)
    DICE = DiceLoss()

    def __str__(self):
        """Inheritance str method."""
        return self.name

    def __call__(self, *args, **kwargs):
        """Inheritance call method."""
        return self.value(*args, **kwargs)

    def list_options():
        """Generate a list of available loss function options for the model.

        This function returns a list of tuples where each tuple contains
        the loss function value and its corresponding string representation.

        Returns
        -------
        zip
            A zip object containing tuples of loss function values and their names.
        """
        return zip(
            [loss.value for loss in Loss], ["ACURACY", "FSCORE", "BCE", "IOU", "IOU_STABLE", "DICE"]
        )

    def list_training_options():
        """Generate a list of training-specific loss function options.

        This function returns a list of tuples containing the loss function values
        and their corresponding names, starting from the third item in the Loss list.

        Returns
        -------
        zip
            A zip object containing tuples of loss function values and their names for training.
        """
        return zip([loss.value for loss in list(Loss)[2:]], ["BCE", "IOU", "IOU_STABLE", "DICE"])


def get_loss_name(loss: Loss):
    """Return the string representation of a given loss function.

    This function takes a `Loss` enum value as input and returns the corresponding
    string name of the loss function.

    Args:
    loss : Loss
        The loss function enum value.

    Returns:
    str
        The string name of the loss function.

    Raises
    ------
    ValueError
        If the provided loss function is not recognized.
    """
    if loss == Loss.ACURACY:
        return "ACURACY"
    elif loss == Loss.FSCORE:
        return "FSCORE"
    elif loss == Loss.CROSS_ENTROPY:
        return "BCE"
    elif loss == Loss.IOU_NORMAL:
        return "IOU"
    elif loss == Loss.IOU_STABLE:
        return "IOU_STABLE"
    elif loss == Loss.DICE:
        return "DICE"
    else:
        ValueError(f"Unknown LOSS: {loss}")
