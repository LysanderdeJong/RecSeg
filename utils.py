import os
import numpy as np
import torch
from einops import rearrange
import wandb

from dataloader import (
    SKMDataModule,
    BrainDWIDataModule,
    TecFideraDataModule,
    TecFideraMRIDataModule,
)

from model.unet import UnetModule, LamdaUnetModule
from model.unetrecon import UnetReconModule
from model.vnet import VnetModule
from model.deeplab import DeepLabModule
from model.unet3d import Unet3dModule
from model.attunet import AttUnetModule
from model.cirim import CIRIMModule
from model.idslr import IDSLRModule
from pl_model import RecSegModule


def segmentation_volume_to_img(seg):
    if len(seg.shape) == 4:
        c_dim = 1
    elif len(seg.shape) == 3:
        c_dim = 0
    else:
        raise ValueError

    if isinstance(seg, np.ndarray):
        img = np.argmax(seg, axis=c_dim)
        img = rearrange(img, "c h w -> h w c")
    elif isinstance(seg, torch.Tensor):
        img = torch.argmax(seg, dim=c_dim)
    return img


def get_model(parser=None, args=None, model=None, **kwargs):
    if parser and args:
        if args.model == "unet":
            parser = UnetModule.add_model_specific_args(parser)
        elif args.model == "unetrecon":
            parser = UnetReconModule.add_model_specific_args(parser)
        elif args.model == "lambdaunet":
            parser = LamdaUnetModule.add_model_specific_args(parser)
        elif args.model == "vnet":
            parser = VnetModule.add_model_specific_args(parser)
        elif args.model == "deeplab":
            parser = DeepLabModule.add_model_specific_args(parser)
        elif args.model == "unet3d":
            parser = Unet3dModule.add_model_specific_args(parser)
        elif args.model == "attunet":
            parser = AttUnetModule.add_model_specific_args(parser)
        elif args.model == "cirim":
            parser = CIRIMModule.add_model_specific_args(parser)
        elif args.model == "recseg":
            parser = RecSegModule.add_model_specific_args(parser)
        elif args.model == "idslr":
            parser = IDSLRModule.add_model_specific_args(parser)
        else:
            raise NotImplementedError
        return parser
    elif model:
        if model == "unet":
            return UnetModule(**kwargs)
        elif model == "unetrecon":
            return UnetReconModule(**kwargs)
        elif model == "lambdaunet":
            return LamdaUnetModule(**kwargs)
        elif model == "vnet":
            return VnetModule(**kwargs)
        elif model == "deeplab":
            return DeepLabModule(**kwargs)
        elif model == "unet3d":
            return Unet3dModule(**kwargs)
        elif model == "attunet":
            return AttUnetModule(**kwargs)
        elif model == "cirim":
            return CIRIMModule(**kwargs)
        elif model == "recseg":
            return RecSegModule(**kwargs)
        elif model == "idslr":
            return IDSLRModule(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise ValueError


def get_dataset(parser=None, args=None, dataset=None, **kwargs):
    if parser and args:
        if args.dataset == "skmtea":
            parser = SKMDataModule.add_data_specific_args(parser)
        elif args.dataset == "braindwi":
            parser = BrainDWIDataModule.add_data_specific_args(parser)
        elif args.dataset == "tecfidera":
            parser = TecFideraDataModule.add_data_specific_args(parser)
        elif args.dataset == "tecfideramri":
            parser = TecFideraMRIDataModule.add_data_specific_args(parser)
        else:
            raise NotImplementedError
        return parser
    elif dataset:
        if dataset == "skmtea":
            return SKMDataModule(**kwargs)
        elif dataset == "braindwi":
            return BrainDWIDataModule(**kwargs)
        elif dataset == "tecfidera":
            return TecFideraDataModule(**kwargs)
        elif dataset == "tecfideramri":
            return TecFideraMRIDataModule(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise ValueError


def retrieve_checkpoint(
    model_id, project="techfidera-recseg", epoch="best", download_dir=None
):
    api = wandb.Api()
    artifact_path = os.path.join("lysander", project, f"model-{model_id}:{epoch}")
    artifact = api.artifact(artifact_path, type="model")
    return artifact.get_path("model.ckpt").download(root=download_dir)


import sys
import inspect
import argparse
from jsonargparse import Namespace
from jsonargparse.util import ParserError, _lenient_check_context
from jsonargparse.typehints import ActionTypeHint
from jsonargparse.loaders_dumpers import load_value_context
from unittest.mock import patch


def parse_known_args(self, args=None, namespace=None):
    """Raises NotImplementedError to dissuade its use, since typos in configs would go unnoticed."""
    caller = inspect.getmodule(inspect.stack()[1][0]).__package__
    # if caller not in {'jsonargparse', 'argcomplete'}:
    #     raise NotImplementedError('parse_known_args not implemented to dissuade its use, since typos in configs would go unnoticed.')
    if args is None:
        args = sys.argv[1:]
    else:
        args = list(args)
        if not all(isinstance(a, str) for a in args):
            self.error(f"All arguments are expected to be strings: {args}")

    if namespace is None:
        namespace = Namespace()

    if caller == "argcomplete":
        namespace.__class__ = Namespace
        namespace = self.merge_config(
            self.get_defaults(skip_check=True), namespace
        ).as_flat()

    try:
        with patch("argparse.Namespace", Namespace), _lenient_check_context(
            caller
        ), ActionTypeHint.subclass_arg_context(self), load_value_context(
            self.parser_mode
        ):
            namespace, args = self._parse_known_args(args, namespace)
    except (argparse.ArgumentError, ParserError) as ex:
        self.error(str(ex), ex)

    return namespace, args
