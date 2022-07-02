from collections import defaultdict
import os
import h5py
import jsonargparse
import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar

from utils import parse_known_args, get_dataset, get_model, retrieve_checkpoint


def setup(args):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args.seed = pl.seed_everything(args.seed, workers=True)
    if args.out_dir is None:
        raise ValueError
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Create a PyTorch Lightning trainer
    callbacks = []
    callbacks.append(ModelSummary(max_depth=2))
    callbacks.append(TQDMProgressBar(refresh_rate=1 if args.progress_bar else 0))

    checkpoint_path = (
        retrieve_checkpoint(model_id=args.model_id)
        if args.model_id is not None
        else args.checkpoint_path,
    )[0]

    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        auto_select_gpus=True,
        gpus=args.gpus,
        max_epochs=args.epochs,
        callbacks=callbacks,
        auto_scale_batch_size="binsearch" if args.auto_batch else None,
        auto_lr_find=True if args.auto_lr else False,
        precision=args.precision,
        limit_train_batches=args.train_limit,
        limit_val_batches=args.val_limit,
        limit_test_batches=args.test_limit,
        accumulate_grad_batches=args.grad_batch,
        gradient_clip_val=args.grad_clip,
        benchmark=True if args.benchmark else False,
        enable_model_summary=False,
        fast_dev_run=True if args.fast_dev_run else False,
    )
    trainer.logger._default_hp_metric = None
    trainer.logger._log_graph = False

    dict_args = vars(args)
    # Create dataloaders
    pl_loader = get_dataset(**dict_args)
    # Create model
    model = get_model(**dict_args)

    if checkpoint_path:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=model.device)["state_dict"],
            strict=False,
        )
    trainer.test(model=model, datamodule=pl_loader)


if __name__ == "__main__":
    print("Check the Tensorboard to monitor training progress.")
    jsonargparse.ArgumentParser.parse_known_args = parse_known_args
    parser = jsonargparse.ArgumentParser()

    # figure out which model to use
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)

    temp_args, _ = parser.parse_known_args()

    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)

    parser = get_model(parser=parser, args=temp_args)
    parser = get_dataset(parser=parser, args=temp_args)

    # trainer hyperparameters
    parser.add_argument(
        "--epochs", default=200, type=int, help="Number of epochs to train."
    )

    parser.add_argument(
        "--precision",
        default=32,
        type=int,
        choices=["16", "32", "64", "bf16"],
        help="At what precision the model should train.",
    )
    parser.add_argument(
        "--grad_batch",
        default=1,
        type=int,
        help="Accumulate gradient to simulate larger batch sizes.",
    )
    parser.add_argument(
        "--grad_clip", default=None, type=float, help="Clip the gradient norm."
    )

    parser.add_argument(
        "--gpus", nargs="+", default=None, type=int, help="Which gpus to use."
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Continue training from this checkpoint.",
    )

    parser.add_argument(
        "--model_id", default=None, type=str, help="Retieve the checkpoint from wandb.",
    )

    parser.add_argument(
        "--out_dir", default=None, type=str, help="Path to save the results",
    )

    parser.add_argument(
        "--eval_mode",
        default=None,
        type=str,
        choices=["recon", "segmentation", "srs"],
        help="Path to save the results",
    )

    parser.add_argument(
        "--mc_samples", default=1, type=int, help="Number of MC dropout samples.",
    )

    # other hyperparameters
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--log_dir",
        default="logs/",
        type=str,
        help="Directory where the PyTorch Lightning logs " + "should be created.",
    )

    parser.add_argument(
        "--train_limit", default=1.0, type=float, help="Percentage of data to train on."
    )
    parser.add_argument(
        "--val_limit",
        default=1.0,
        type=float,
        help="Percentage of data to validate with.",
    )
    parser.add_argument(
        "--test_limit", default=1.0, type=float, help="Percentage of data to test with."
    )

    parser.add_argument(
        "--progress_bar",
        action="store_true",
        help="Use a progress bar indicator for interactive experimentation. "
        + "Not to be used in conjuction with SLURM jobs.",
    )
    parser.add_argument(
        "--auto_lr",
        action="store_true",
        help="When used tries to automatically set an appropriate learning rate.",
    )
    parser.add_argument(
        "--auto_batch",
        action="store_true",
        help="When used tries to automatically set an appropriate batch size.",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Enables cudnn auto-tuner."
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enables logging to wandb, otherwise uses tensorbaord.",
    )

    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Runs a single batch for train, val and test.",
    )

    parser.add_argument(
        "--profiler",
        default=None,
        type=str,
        choices=["simple", "advanced", "pytorch"],
        help="Code profiler.",
    )

    args = parser.parse_args()
    # print(args)
    setup(args)
