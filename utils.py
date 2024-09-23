import random
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.nn import BCEWithLogitsLoss
from torchvision.transforms.v2 import functional as F


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, epsilon=1e-6, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    target = target.float()
    fn = multiclass_dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index, epsilon=epsilon)


class MyDiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 loss_weight=1.0,
                 ignore_index=-100,
                 eps=1e-3,
                 loss_name='my_loss_dice'):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_dice'.
        """

        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.eps = eps
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        one_hot_target = target
        if (pred.shape != target.shape):
            one_hot_target = one_hot_target.unsqueeze(1)

        if self.use_sigmoid:
            pred = pred.sigmoid()

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            epsilon=self.eps,
            ignore_index=self.ignore_index)

        return loss


@torch.no_grad()
def MaxMinScaler(img: torch.Tensor, target: torch.Tensor):
    """MaxMinScaler for every img
       img: shape like [c,*,*]"""
    min = img.min(1, True).values.min(2, True).values
    max = img.max(1, True).values.max(2, True).values
    img = (img - min) / (max - min)
    return img, target


def get_transform(img, target, *,
                  mean, std,
                  pv=0.5, ph=0.5
                  ):
    img = F.normalize(img, mean, std)
    if random.random() < pv:
        img = F.vertical_flip_image(img)
        target = F.vertical_flip_image(target)
    if random.random() < ph:
        img = F.horizontal_flip(img)
        target = F.horizontal_flip(target)
    return img, target


class LossFunc():
    def __init__(self, bce_weight=0.5, ignore_index=15):
        self.dice_f = MyDiceLoss(ignore_index=15)
        self.bce_f = BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.ignore_index = ignore_index

    def __call__(self, pred: torch.tensor, label: torch.tensor):
        roi = (label != self.ignore_index)
        dice = self.dice_f(pred, label)
        bce = self.bce_f(pred[roi], label[roi])
        return dice + self.bce_weight*bce


def train_one_epoch(model, optimizer, data_loader, device, epoch
                    ) -> tuple[float, float, float, float]:
    model.train()
    start = time.time()
    loss_f = LossFunc()
    loss_total = 0.

    for image, mask in tqdm(data_loader, desc=f"train epoch: {epoch}"):
        image, mask = image.to(device), mask.to(device)
        output = model(image)  # dict
        loss = loss_f(output, mask)
        loss_total += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024.0 ** 3)  # GB
        torch.cuda.reset_peak_memory_stats()
    epoch_time = (time.time() - start)/60  # minutes
    loss_ave = loss_total / len(data_loader)
    aux_ce = -1

    return loss_ave, aux_ce, mem, epoch_time


def val_one_epoch(model, data_loader, device, ignore_index=-100
                  ) -> tuple[np.ndarray, float, float]:
    model.eval()
    with torch.no_grad():
        start = time.time()
        confmat = np.zeros((2, 2))
        diceloss = MyDiceLoss(use_sigmoid=False, ignore_index=ignore_index)
        dice = 0

        for image, mask in tqdm(data_loader, desc="valid epoch: "):
            image, mask = image.to(device), mask.to(device)
            output = model(image)
            predict = torch.where(output.sigmoid() > 0.5, 1., 0.)
            dice += diceloss(predict, mask)
            predict = predict.cpu().numpy().astype(np.uint8).flatten()
            mask = mask.cpu().numpy().astype(np.uint8).flatten()
            if ignore_index >= 0:
                ig_index = (mask == ignore_index)
                predict = predict[~ig_index]
                mask = mask[~ig_index]
            confmat += confusion_matrix(mask, predict, labels=[0, 1])

        dice = dice/len(data_loader)
        epoch_time = (time.time() - start)/60  # minutes
    return confmat, epoch_time, dice.item()


def val_baige(model, data_loader, device, ignore_index=-100
              ) -> tuple[np.ndarray, float, float]:
    model.eval()
    with torch.no_grad():
        start = time.time()
        confmat = np.zeros((2, 2))
        diceloss = MyDiceLoss(use_sigmoid=False, ignore_index=ignore_index)
        dice = 0

        for image, mask in tqdm(data_loader, desc="valid epoch: "):
            image, mask = image.to(device), mask.to(device)
            h, w = image.shape[-2:]
            new_h = (h // 32 + 1) * 32 if h % 32 != 0 else h
            new_w = (w // 32 + 1) * 32 if w % 32 != 0 else w
            pad_shape = ((new_w-w)//2,
                         (new_h-h)//2,
                         new_w-w-(new_w-w)//2,
                         new_h-h-(new_h-h)//2,)
            image = image.to(device, dtype=torch.float)
            image = F.pad(image, pad_shape)
            image = image.to(device)
            output = model(image)
            predict = torch.where(output.sigmoid() > 0.5, 1., 0.)
            predict = predict[..., pad_shape[1]:pad_shape[1]+h, pad_shape[0]:pad_shape[0]+w]

            dice += diceloss(predict, mask)
            predict = predict.cpu().numpy().astype(np.uint8).flatten()
            mask = mask.cpu().numpy().astype(np.uint8).flatten()
            if ignore_index >= 0:
                ig_index = (mask == ignore_index)
                predict = predict[~ig_index]
                mask = mask[~ig_index]
            confmat += confusion_matrix(mask, predict, labels=[0, 1])

        dice = dice/len(data_loader)
        epoch_time = (time.time() - start)/60  # minutes
    return confmat, epoch_time, dice.item()
