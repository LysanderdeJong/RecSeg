import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
from torchmetrics import functional as FM

from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseSensitivityModel
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest


class ZF(nn.Module):
    def __init__(
        self,
        zf_method="SENSE",
        use_sense_net=False,
        sens_chans=8,
        sens_pools=4,
        sens_mask_type="2D",
        fft_type="orthogonal",
    ):
        super().__init__()

        self.fft_type = fft_type
        self.zf_method = zf_method

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = use_sense_net
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                sens_chans,
                sens_pools,
                fft_type=self.fft_type,
                mask_type=sens_mask_type,
            )

    @staticmethod
    def process_inputs(y, mask):
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        return y, mask, r

    def forward(
        self,
        y,
        sensitivity_maps,
        mask,
        init_pred,
        target,
    ):
        sensitivity_maps = (
            self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        )
        eta = coil_combination(
            ifft2c(y, fft_type=self.fft_type),
            sensitivity_maps,
            method=self.zf_method.upper(),
            dim=1,
        )
        eta = torch.view_as_complex(eta)
        _, eta = center_crop_to_smallest(target, eta)
        return eta


class ZFModule(pl.LightningModule):
    def __init__(
        self,
        zf_method="SENSE",
        use_sense_net=False,
        sens_chans=8,
        sens_pools=4,
        sens_mask_type="2D",
        fft_type="orthogonal",
        lr=0.001,
        weight_decay=0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.model = ZF(
            zf_method=self.hparams.zf_method,
            use_sense_net=self.hparams.use_sense_net,
            sens_chans=self.hparams.sens_chans,
            sens_pools=self.hparams.sens_pools,
            sens_mask_type=self.hparams.sens_mask_type,
            fft_type=self.hparams.fft_type,
        )
        self.example_input_array = [
            torch.rand(1, 32, 200, 200, 2),  # kspace
            torch.rand(1, 32, 200, 200, 2),  # sesitivity maps
            torch.rand(1, 1, 200, 200, 1),  # mask
            torch.rand(1, 200, 200, 2),  # initial prediction
            torch.rand(1, 200, 200, 2),  # target
        ]

    def forward(
        self,
        y,
        sensitivity_maps,
        mask,
        init_pred,
        target,
    ):

        return self.model(y, sensitivity_maps, mask, init_pred, target)

    def step(self, batch, batch_indx=None):
        (
            y,
            sensitivity_maps,
            mask,
            init_pred,
            target,
            fname,
            slice_num,
            acc,
            segmentation,
        ) = batch
        y, mask, _ = self.model.process_inputs(y, mask)

        y = self.fold(y)
        sensitivity_maps = self.fold(sensitivity_maps)
        mask = self.fold(mask)
        target = self.fold(target)

        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)
        # preds = preds / torch.abs(preds).amax((-1, -2), True)
        # print(torch.abs(preds).max(), torch.abs(target).max())

        # if torch.any(torch.isnan(preds[-1][-1])):
        #     print(preds[-1][-1])
        #     raise ValueError

        output = torch.abs(preds) / torch.abs(preds).amax((-1, -2), True)
        target = torch.abs(target) / torch.abs(target).amax((-1, -2), True)

        loss_dict = {
            "loss": F.l1_loss(output.unsqueeze(-3), target.unsqueeze(-3)),
            "l1": F.l1_loss(output.unsqueeze(-3), target.unsqueeze(-3)),
            "psnr": FM.psnr(output.unsqueeze(-3), target.unsqueeze(-3)),
            "ssim": FM.ssim(output.unsqueeze(-3), target.unsqueeze(-3)),
        }
        return loss_dict, preds

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return loss_dict, output

    def test_step(self, batch, batch_idx):
        fname = batch[5]
        slice_num = batch[6]
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"test_{metric}", value.mean().detach(), sync_dist=True)
        return loss_dict, (fname, slice_num, output.detach().cpu().numpy())

    # def test_epoch_end(self, outputs):
    #     reconstructions = defaultdict(list)
    #     for fname, slice_num, output in outputs[1]:
    #         reconstructions[fname].append((slice_num, output))

    #     for fname in reconstructions:
    #         reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])  # type: ignore

    #     out_dir = Path(os.path.join(self.logger.log_dir, "reconstructions"))
    #     out_dir.mkdir(exist_ok=True, parents=True)
    #     for fname, recons in reconstructions.items():
    #         with h5py.File(out_dir / fname, "w") as hf:
    #             hf.create_dataset("reconstruction", data=recons)

    def predict_step(self, batch, batch_idx):
        fname = batch[5]
        slice_num = batch[6]
        max_value = batch[7][-1][0]
        loss_dict, output = self.step(batch, batch_idx)
        output = (output * max_value).unsqueeze(0).detach().cpu()
        return loss_dict, (fname, slice_num, output)

    def fold(self, tensor):
        shape = list(tensor.shape[1:])
        shape[0] = -1
        return tensor.view(shape)

    def unfold(self, tensor):
        shape = list(1, tensor.shape)
        shape[1] = -1
        return tensor.view(shape)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="min", factor=0.5, patience=4, cooldown=1
            ),
            "monitor": "val_loss",
        }

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ZFModel")

        # network params
        parser.add_argument(
            "--zf_method",
            choices=["SENSE", "RSS"],
            default="SENSE",
            help="Type of output to use",
        )
        parser.add_argument(
            "--use_sens_net",
            action="store_true",
            default=False,
            help="Use sensitivity net",
        )
        parser.add_argument(
            "--sens_chans",
            type=int,
            default=8,
            help="Number of channels for the sensitivity net",
        )
        parser.add_argument(
            "--sens_pools",
            type=int,
            default=4,
            help="Number of pools for the sensitivity net",
        )
        parser.add_argument(
            "--sens_mask_type",
            choices=["1D", "2D"],
            default="2D",
            help="Type of mask to use for the sensitivity net",
        )

        parser.add_argument(
            "--fft_type", type=str, default="normal", help="Type of FFT to use"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=1e-3, type=float, help="Optimizer learning rate"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        return parent_parser
