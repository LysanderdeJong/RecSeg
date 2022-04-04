import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
                pred, sensitivity_maps, method=self.output_type, dim=1
            )
        pred = torch.view_as_complex(pred)
        _, pred = center_crop_to_smallest(target, pred)
        return pred

    @staticmethod
    def process_input(y, mask):
        """Process the inputs to the network."""
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        return y, mask, r

    def calculate_loss(self, eta, target, _loss_fn=F.l1_loss):
        target = torch.abs(target / torch.max(torch.abs(target)))
        if not self.accumulate_estimates:
            return _loss_fn(torch.abs(eta / torch.max(torch.abs(eta))), target)
        cascade_loss = []
        for cascade_eta in eta:
            time_step_loss = [
                _loss_fn(
                    torch.abs(time_step_eta / torch.max(torch.abs(time_step_eta))),
                    target,
                )
                for time_step_eta in cascade_eta
            ]
            time_step_loss = torch.stack(time_step_loss)
            loss_weights = torch.logspace(
                -1, 0, steps=len(time_step_loss), device=time_step_loss.device
            )
            cascade_loss.append(
                sum(time_step_loss * loss_weights) / len(time_step_loss)
            )
        return sum(cascade_loss) / len(cascade_loss)
