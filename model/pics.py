import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
from torchmetrics import functional as FM

try:
    import bart
except Exception:
    pass

from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseSensitivityModel
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest


class PICS(nn.Module):
    def __init__(
        self,
        reg_wt=0.005,
        num_iters=60,
        device="cuda",
        use_sense_net=False,
        sens_chans=8,
        sens_pools=4,
        sens_mask_type="2D",
        output_type="SENSE",
        fft_type="orthogonal",
    ):
        super().__init__()

        self.reg_wt = reg_wt
        self.num_iters = num_iters
        self._device = device
        self.fft_type = fft_type
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
        target,
    ):
        sensitivity_maps = (
            self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        )
        if "cuda" in str(self._device):
            eta = bart.bart(
                1,
                f"pics -d0 -g -S -R W:7:0:{self.reg_wt} -i {self.num_iters}",
                y,
                sensitivity_maps,
            )[0]
        else:
            eta = bart.bart(
                1,
                f"pics -d0 -S -R W:7:0:{self.reg_wt} -i {self.num_iters}",
                y,
                sensitivity_maps,
            )[0]
        _, eta = center_crop_to_smallest(target, eta)
        return eta


class PICSModule(pl.LightningModule):
    def __init__(
        self,
        reg_wt=0.005,
        num_iters=60,
        device_str="cuda",
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

        self.model = PICS(
            reg_wt=self.hparams.reg_wt,
            num_iters=self.hparams.num_iters,
            device=self.hparams.device_str,
            use_sense_net=self.hparams.use_sense_net,
            sens_chans=self.hparams.sens_chans,
            sens_pools=self.hparams.sens_pools,
            sens_mask_type=self.hparams.sens_mask_type,
            fft_type=self.hparams.fft_type,
            output_type="SENSE",
        )
        self.example_input_array = [
            torch.rand(1, 32, 200, 200, 2),  # kspace
            torch.rand(1, 32, 200, 200, 2),  # sesitivity maps
            torch.rand(1, 1, 200, 200, 1),  # mask
            torch.rand(1, 200, 200, 2),  # target
        ]

    def forward(
        self,
        y,
        sensitivity_maps,
        mask,
        target,
    ):
        return self.model(y, sensitivity_maps, mask, target)

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

        y = torch.view_as_complex(y)
        # y = torch.fft.fftshift(y, dim=[-2, -1])
        # if self.hparams.fft_type != "orthogonal":
        #     y = torch.fft.fftshift(y, dim=(-2, -1))
        y = y.permute(0, 2, 3, 1).detach().cpu().numpy()

        if sensitivity_maps is None and not self.sens_net:
            raise ValueError(
                "Sensitivity maps are required for PICS. "
                "Please set use_sens_net to True if you precomputed sensitivity maps are not available."
            )

        sensitivity_maps = torch.view_as_complex(sensitivity_maps)
        sensitivity_maps = sensitivity_maps / torch.amax(
            torch.abs(sensitivity_maps), keepdim=True
        )
        # sensitivity_maps = torch.fft.fft2(sensitivity_maps, dim=[-2, -1])
        # sensitivity_maps = torch.fft.fftshift(sensitivity_maps, dim=[-2, -1])
        # sensitivity_maps = torch.fft.ifft2(sensitivity_maps, dim=[-2, -1])
        if self.hparams.fft_type != "orthogonal":
            sensitivity_maps = torch.fft.fftshift(sensitivity_maps, dim=(-2, -1))
        sensitivity_maps = sensitivity_maps.permute(0, 2, 3, 1).detach().cpu().numpy()  # type: ignore

        preds = torch.from_numpy(
            self.forward(y, sensitivity_maps, mask, target)
        ).unsqueeze(0)
        if self.hparams.fft_type != "orthogonal":
            preds = torch.fft.fftshift(preds, dim=(-2, -1))

        output = torch.abs(preds) / torch.abs(preds).amax((-1, -2), True)
        target = torch.abs(target) / torch.abs(target).amax((-1, -2), True)

        target = target.to(output.device)

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
        parser = parent_parser.add_argument_group("PICSmodel")

        # network params
        parser.add_argument(
            "--reg_wt",
            type=float,
            default=0.005,
            help="Regularization strenght",
        )
        parser.add_argument(
            "--num_iters",
            type=int,
            default=60,
            help="Number of iterationts do PCIS reconstruction.",
        )
        parser.add_argument(
            "--device_str",
            type=str,
            default="cuda",
            help="Device on which to perform the reconstruction.",
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
