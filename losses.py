import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        include_background = True,
        squared_pred = False,
        smooth_nr = 1e-5,
        smooth_dr = 1e-5,
        batch = False):
        super().__init__()
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch


    def forward(self, input, target):
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        
        input = torch.softmax(input, dim=1)
            
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
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

        f = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        f = torch.mean(f)
        return f


class DiceCELoss(nn.Module):
    def __init__(
        self,
        include_background = True,
        squared_pred = False,
        smooth_nr = 1e-5,
        smooth_dr = 1e-5,
        batch = False,
        ce_weight = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight)
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce


    def ce(self, input, target):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        return self.cross_entropy(input, target)


    def forward(self, input, target):
        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        return total_loss