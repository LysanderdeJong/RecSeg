import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import functional as FM

from mridc.collections.reconstruction.models.rim.rim_block import RIMBlock
from mridc.collections.reconstruction.models.base import BaseSensitivityModel
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.rnn_utils import rnn_weights_init
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest


class CIRIM(nn.Module):
    def __init__(
        self,
        recurrent_layer: str = "IndRNN",
        conv_filters=None,
        conv_kernels=None,
        conv_dilations=None,
        conv_bias=None,
        recurrent_filters=None,
        recurrent_kernels=None,
        recurrent_dilations=None,
        recurrent_bias=None,
        depth=2,
        time_steps=8,
        conv_dim=2,
        num_cascades=8,
        no_dc=True,
        keep_eta=True,
        use_sens_net=False,
        sens_chans=8,
        sens_pools=4,
        sens_mask_type="2D",
        fft_type="orthogonal",
        output_type="SENSE",
    ):
        if conv_filters is None:
            conv_filters = [64, 64, 2]
        if conv_kernels is None:
            conv_kernels = [5, 3, 3]
        if conv_dilations is None:
            conv_dilations = [1, 2, 1]
        if conv_bias is None:
            conv_bias = [True, True, False]
        if recurrent_filters is None:
            recurrent_filters = [64, 64, 0]
        if recurrent_kernels is None:
            recurrent_kernels = [1, 1, 0]
        if recurrent_dilations is None:
            recurrent_dilations = [1, 1, 0]
        if recurrent_bias is None:
            recurrent_bias = [True, True, False]
        super().__init__()

        assert (
            len(conv_filters)
            == len(conv_kernels)
            == len(conv_dilations)
            == len(conv_bias)
            == len(recurrent_filters)
            == len(recurrent_kernels)
            == len(recurrent_dilations)
            == len(recurrent_bias)
        )

        self.time_steps = 8 * math.ceil(time_steps / 8)
        self.no_dc = no_dc
        self.fft_type = fft_type
        self.num_cascades = num_cascades

        self.cirim = torch.nn.ModuleList(
            [
                RIMBlock(
                    recurrent_layer=recurrent_layer,
                    conv_filters=conv_filters,
                    conv_kernels=conv_kernels,
                    conv_dilations=conv_dilations,
                    conv_bias=conv_bias,
                    recurrent_filters=recurrent_filters,
                    recurrent_kernels=recurrent_kernels,
                    recurrent_dilations=recurrent_dilations,
                    recurrent_bias=recurrent_bias,
                    depth=depth,
                    time_steps=self.time_steps,
                    conv_dim=conv_dim,
                    no_dc=self.no_dc,
                    fft_type=self.fft_type,
                )
                for _ in range(self.num_cascades)
            ]
        )

        self.keep_eta = keep_eta
        self.output_type = output_type

        self.use_sens_net = use_sens_net
        if self.use_sens_net:
            self.sense_net = BaseSensitivityModel(
                sens_chans, sens_pools, fft_type=self.fft_type, mask_type=sens_mask_type
            )

        std_init_range = 1 / recurrent_filters[0] ** 0.5
        self.cirim.apply(lambda module: rnn_weights_init(module, std_init_range))

        # self.dc_weight = torch.nn.Parameter(torch.ones(1))
        self.accumulate_estimates = True

    def forward(self, y, sensitivity_maps, mask, init_pred, target):
        sensitivity_maps = (
            self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        )
        prediction = y.clone()
        init_pred = None if init_pred is None or init_pred.dim() < 4 else init_pred
        hx = None
        sigma = 1.0
        cascades_etas = []
        for i, cascade in enumerate(self.cirim):
            # Forward pass through the cascades
            prediction, hx = cascade(
                prediction,
                y,
                sensitivity_maps,
                mask,
                init_pred,
                hx,
                sigma,
                keep_eta=False if i == 0 else self.keep_eta,
            )
            time_steps_etas = [
                self.process_intermediate_pred(pred, sensitivity_maps, target)
                for pred in prediction
            ]
            cascades_etas.append(time_steps_etas)
        return cascades_etas

    def process_intermediate_pred(
        self, pred, sensitivity_maps, target, do_coil_combination=False
    ):
        """Process the intermediate eta to be used in the loss function."""
        # Take the last time step of the eta
        if not self.no_dc or do_coil_combination:
            pred = ifft2c(pred, fft_type=self.fft_type)
            pred = coil_combination(
                pred, sensitivity_maps, method=self.output_type, dim=-4
            )
        pred = torch.view_as_complex(pred)
        _, pred = center_crop_to_smallest(target, pred)
        return pred

    @staticmethod
    def process_inputs(y, mask):
        """Process the inputs to the network."""
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        return y, mask, r

    def calculate_loss(self, eta, target, _loss_fn=F.l1_loss):
        target = torch.abs(target / torch.abs(target).amax((-1, -2), True))
        if not self.accumulate_estimates:
            return _loss_fn(
                torch.abs(eta / torch.abs(eta).amax((-1, -2), True)), target
            )

        cascades_loss = []
        for cascade_eta in eta:
            time_step_loss = [
                _loss_fn(
                    torch.abs(
                        time_step_eta / torch.abs(time_step_eta).amax((-1, -2), True)
                    ),
                    target,
                )
                for time_step_eta in cascade_eta
            ]
            time_step_loss_stack = torch.stack(time_step_loss)
            loss_weights = torch.logspace(
                -1, 0, steps=len(time_step_loss), device=time_step_loss_stack.device
            )
            cascades_loss.append(
                sum(time_step_loss_stack * loss_weights) / len(time_step_loss)
            )
        return sum(cascades_loss) / len(cascades_loss)


class CIRIMModule(pl.LightningModule):
    def __init__(
        self,
        recurrent_layer: str = "IndRNN",
        conv_filters=None,
        conv_kernels=None,
        conv_dilations=None,
        conv_bias=None,
        recurrent_filters=None,
        recurrent_kernels=None,
        recurrent_dilations=None,
        recurrent_bias=None,
        depth: int = 2,
        time_steps: int = 8,
        conv_dim: int = 2,
        loss_fn: str = F.l1_loss,
        num_cascades: int = 1,
        no_dc: bool = False,
        keep_eta: bool = False,
        use_sens_net: bool = False,
        sens_chans: int = 8,
        sens_pools: int = 4,
        sens_mask_type: str = "2D",
        fft_type: str = "orthogonal",
        output_type: str = "SENSE",
        lr=0.001,
        weight_decay=0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.model = CIRIM(
            recurrent_layer=self.hparams.recurrent_layer,
            conv_filters=self.hparams.conv_filters,
            conv_kernels=self.hparams.conv_kernels,
            conv_dilations=self.hparams.conv_dilations,
            conv_bias=self.hparams.conv_bias,
            recurrent_filters=self.hparams.recurrent_filters,
            recurrent_kernels=self.hparams.recurrent_kernels,
            recurrent_dilations=self.hparams.recurrent_dilations,
            recurrent_bias=self.hparams.recurrent_bias,
            depth=self.hparams.depth,
            time_steps=self.hparams.time_steps,
            conv_dim=self.hparams.conv_dim,
            num_cascades=self.hparams.num_cascades,
            no_dc=self.hparams.no_dc,
            keep_eta=self.hparams.keep_eta,
            use_sens_net=self.hparams.use_sens_net,
            sens_chans=self.hparams.sens_chans,
            sens_pools=self.hparams.sens_pools,
            sens_mask_type=self.hparams.sens_mask_type,
            fft_type=self.hparams.fft_type,
            output_type=self.hparams.output_type,
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
        loss = self.model.calculate_loss(preds, target, _loss_fn=self.hparams.loss_fn)

        # if torch.any(torch.isnan(preds[-1][-1])):
        #     print(preds[-1][-1])
        #     raise ValueError

        output = torch.abs(preds[-1][-1])
        output = output / output.amax((-1, -2), True)
        target = torch.abs(target) / torch.abs(target).amax((-1, -2), True)

        loss_dict = {
            "loss": loss,
            "l1": loss,
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
        if isinstance(output, list):
            output = output[-1]
        if isinstance(output, list):
            output = output[-1]
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
        if isinstance(output, list):
            output = [i.unsqueeze(0).detach().cpu() for j in output for i in j]
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
        parser = parent_parser.add_argument_group("CIRIMModel")

        # network params
        parser.add_argument(
            "--num_cascades",
            type=int,
            default=1,
            help="Number of cascades for the model",
        )
        parser.add_argument(
            "--time_steps", type=int, default=8, help="Number of RIM steps"
        )
        parser.add_argument(
            "--recurrent_layer", type=str, default="IndRNN", help="Recurrent layer type"
        )
        parser.add_argument(
            "--conv_filters",
            nargs="+",
            default=[64, 64, 2],
            type=int,
            help="Number of filters the convolutional layers of for the model",
        )
        parser.add_argument(
            "--conv_kernels",
            nargs="+",
            default=[5, 3, 3],
            type=int,
            help="Kernel size for the convolutional layers of the model",
        )
        parser.add_argument(
            "--conv_dilations",
            nargs="+",
            type=int,
            default=[1, 2, 1],
            help="Dilations for the convolutional layers of the model",
        )
        parser.add_argument(
            "--conv_bias",
            nargs="+",
            type=bool,
            default=[True, True, False],
            help="Bias for the convolutional layers of the model",
        )
        parser.add_argument(
            "--recurrent_filters",
            nargs="+",
            type=int,
            default=[64, 64, 0],
            help="Number of filters the recurrent layers of for the model",
        )
        parser.add_argument(
            "--recurrent_kernels",
            nargs="+",
            type=int,
            default=[1, 1, 0],
            help="Kernel size for the recurrent layers of the model",
        )
        parser.add_argument(
            "--recurrent_dilations",
            nargs="+",
            type=int,
            default=[1, 1, 0],
            help="Dilations for the recurrent layers of the model",
        )
        parser.add_argument(
            "--recurrent_bias",
            nargs="+",
            type=bool,
            default=[True, True, False],
            help="Bias for the recurrent layers of the model",
        )
        parser.add_argument("--depth", type=int, default=2, help="Depth of the model")
        parser.add_argument(
            "--conv_dim",
            type=int,
            default=2,
            help="Dimension of the convolutional layers",
        )
        parser.add_argument(
            "--no_dc",
            action="store_false",
            default=True,
            help="Do not use DC component",
        )
        parser.add_argument(
            "--keep_eta", action="store_false", default=True, help="Keep eta constant"
        )
        parser.add_argument(
            "--use_sens_net",
            action="store_true",
            default=False,
            help="Use sensitivity net",
        )
        parser.add_argument(
            "--sens_pools",
            type=int,
            default=4,
            help="Number of pools for the sensitivity net",
        )
        parser.add_argument(
            "--sens_chans",
            type=int,
            default=8,
            help="Number of channels for the sensitivity net",
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
