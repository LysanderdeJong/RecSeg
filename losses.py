import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        include_background=True,
        squared_pred=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        batch=False,
    ):
        super().__init__()
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input, target):
        if target.shape != input.shape:
            raise AssertionError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )

        input = torch.softmax(input, dim=1)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        dice = (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        loss = 1.0 - dice

        return loss.mean(1), dice.mean(1)


class ContourLoss(nn.Module):
    """https://github.com/rosanajurdi/Perimeter_loss"""

    def forward(self, pred, target):
        pred_contour = self.contour(pred)
        target_contour = self.contour(target)

        loss = (
            pred_contour.flatten(2).sum(-1) - target_contour.flatten(2).sum(-1)
        ).pow(2)
        loss /= pred.shape[-2] * pred.shape[-1]
        return loss.mean()

    def contour(self, x):
        min_pool = F.max_pool2d(x * -1, kernel_size=(3, 3), stride=1, padding=1) * -1
        max_pool = F.max_pool2d(min_pool, kernel_size=(3, 3), stride=1, padding=1)
        x = F.relu(max_pool - min_pool)
        return x
