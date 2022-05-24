import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
from torchmetrics import functional as FM

from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseSensitivityModel
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest


class UNetRecon(nn.Module):
    def __init__(
        self,
        channels=64,
        num_pools=2,
        padding_size=11,
        normalize=True,
        use_sense_net=False,
        sens_chans=8,
        sens_pools=4,
        sens_mask_type="2D",
        output_type="SENSE",
        fft_type="orthogonal",
    ):
        super().__init__()

        self.fft_type = fft_type

        self.unet = NormUnet(
            chans=channels,
            num_pools=num_pools,
            padding_size=padding_size,
            normalize=normalize,
        )

        self.output_type = output_type

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = use_sense_net
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                sens_chans,
                sens_pools,
                fft_type=self.fft_type,
                mask_type=sens_mask_type,
            )

        self.accumulate_estimates = False

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
        self, y, sensitivity_maps, mask, init_pred, target,
    ):
        sensitivity_maps = (
            self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        )
        eta = torch.view_as_complex(
            coil_combination(
                ifft2c(y, fft_type=self.fft_type),
                sensitivity_maps,
                method=self.output_type,
                dim=1,
            )
        )
        _, eta = center_crop_to_smallest(target, eta)
        return torch.view_as_complex(
            self.unet(torch.view_as_real(eta.unsqueeze(1)))
        ).squeeze(1)


class UnetReconModule(pl.LightningModule):
    def __init__(
        self,
        channels=64,
        num_pools=2,
        padding_size=11,
        use_sense_net=False,
        sens_chans=8,
        sens_pools=4,
        sens_mask_type="2D",
        output_type="SENSE",
        fft_type="orthogonal",
        lr=0.001,
        weight_decay=0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.model = UNetRecon(
            channels=self.hparams.channels,
            num_pools=self.hparams.num_pools,
            padding_size=self.hparams.padding_size,
            use_sense_net=self.hparams.use_sense_net,
            sens_chans=self.hparams.sens_chans,
            sens_pools=self.hparams.sens_pools,
            sens_mask_type=self.hparams.sens_mask_type,
            output_type=self.hparams.output_type,
            fft_type=self.hparams.fft_type,
        )
        self.example_input_array = [
            torch.rand(1, 32, 320, 320, 2),  # kspace
            torch.rand(1, 32, 320, 320, 2),  # sesitivity maps
            torch.rand(1, 1, 320, 320, 1),  # mask
            torch.rand(1, 320, 320, 2),  # initial prediction
            torch.rand(1, 320, 320, 2),  # target
        ]

    def forward(
        self, y, sensitivity_maps, mask, init_pred, target,
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
        loss_dict, output = self.step(batch, batch_idx)
        output = output.unsqueeze(0).detach().cpu()
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
        parser = parent_parser.add_argument_group("UnetReconModel")

        # network params
        parser.add_argument(
            "--channels",
            default=64,
            type=int,
            help="Number of top-level U-Net filters.",
        )
        parser.add_argument(
            "--num_pools", default=2, type=int, help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--padding_size", default=11, type=int, help="Size of the padding applied",
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
            "--output_type",
            choices=["SENSE", "RSS"],
            default="SENSE",
            help="Type of output to use",
        )
        parser.add_argument(
            "--fft_type", type=str, default="backward", help="Type of FFT to use"
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
