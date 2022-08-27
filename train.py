import collections
import os
import jsonargparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ModelSummary,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning.profiler import PyTorchProfiler

from callbacks import (
    LogIntermediateReconstruction,
    LogReconstructionTECFIDERA,
    PrintCallback,
    LogCallback,
    NumParamCallback,
    InferenceTimeCallback,
    LogSegmentationMasksSKMTEA,
    LogSegmentationMasksDWI,
    LogSegmentationMasksTECFIDERA,
    LogSegmentationMasksRECSEGTECFIDERA,
    LogUncertaintyTECFIDERA,
)
from utils import parse_known_args, get_dataset, get_model


def train(args):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args.seed = pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create a PyTorch Lightning trainer
    callbacks = []
    modelcheckpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="{epoch}-{val_loss:.4f}",
    )
    callbacks.append(modelcheckpoint)
    # callbacks.append(StochasticWeightAveraging(swa_epoch_start=20, annealing_epochs=10))
    callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=10))
    callbacks.append(ModelSummary(max_depth=2))
    callbacks.append(TQDMProgressBar(refresh_rate=1 if args.progress_bar else 0))
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(NumParamCallback())
    callbacks.append(InferenceTimeCallback())
    if args.wandb:
        os.environ["WANDB_CACHE_DIR"] = "/scratch/lgdejong/.cache/wandb/"
        wandb_logger = WandbLogger(
            project=args.project,
            log_model="all",
            entity="lysander",
        )

        if args.dataset == "skmtea":
            callbacks.append(LogSegmentationMasksSKMTEA())
        elif args.dataset == "braindwi":
            callbacks.append(LogSegmentationMasksDWI())
        elif args.dataset == "tecfidera":
            callbacks.append(LogSegmentationMasksTECFIDERA())
            callbacks.append(LogUncertaintyTECFIDERA())
        elif args.dataset == "tecfideramri":
            callbacks.append(LogReconstructionTECFIDERA())
            callbacks.append(LogIntermediateReconstruction())
        if args.model == "recseg" and args.dataset == "tecfideramri":
            callbacks.append(LogSegmentationMasksRECSEGTECFIDERA())
        elif args.model == "idslr" and args.dataset == "tecfideramri":
            callbacks.append(LogSegmentationMasksRECSEGTECFIDERA())
    else:
        callbacks.append(LogCallback())
    if not args.progress_bar:
        callbacks.append(PrintCallback())

    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        auto_select_gpus=True,
        gpus=None if args.gpus == "None" else args.gpus,
        # strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=args.epochs,
        callbacks=callbacks,
        auto_scale_batch_size="binsearch" if args.auto_batch else None,
        auto_lr_find=True if args.auto_lr else False,
        precision=args.precision,
        limit_train_batches=args.train_limit,
        limit_val_batches=args.val_limit,
        limit_test_batches=args.test_limit,
        accumulate_grad_batches=args.grad_batch,
        resume_from_checkpoint=args.checkpoint_path,
        gradient_clip_val=args.grad_clip,
        benchmark=True if args.benchmark else False,
        # plugins=[MRIDCNativeMixedPrecisionPlugin()],
        # profiler=PyTorchProfiler(
        #     dirpath="/scratch/lgdejong/projects/RecSeg/logs/", filename="recseg_profile"
        # ),  # args.profiler if args.profiler else None,
        enable_model_summary=False,
        logger=wandb_logger if args.wandb else True,
        fast_dev_run=True if args.fast_dev_run else False,
    )
    trainer.logger._default_hp_metric = None
    trainer.logger._log_graph = False

    dict_args = vars(args)
    # Create dataloaders
    pl_loader = get_dataset(**dict_args)
    # Create model
    model = get_model(**dict_args)

    # if not args.progress_bar:
    #     print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
    #           "check the Logger file at " + trainer.logger.log_dir + ". If you " + \
    #           "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Training
    # with torch.autograd.detect_anomaly():
    trainer.tune(model, pl_loader)
    if args.wandb:
        trainer.logger.experiment.watch(model, log="all", log_graph=True)
    trainer.fit(model, pl_loader)
    if args.wandb:
        trainer.logger.experiment.unwatch()
    print(modelcheckpoint.best_model_path)
    trainer.test(model, pl_loader)


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

    # other hyperparameters
    parser.add_argument(
        "--seed", default=None, type=int, help="Seed to use for reproducing results"
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
        "--project",
        default="techfidera-recseg",
        type=str,
        choices=["techfidera-recseg", "dwi-segmentation", "mri-segmentation"],
        help="Wandb project",
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
    train(args)
