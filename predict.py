import collections
from email.policy import strict
import os
import h5py
import jsonargparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from pytorch_lightning.profiler import PyTorchProfiler
from model.cirim import CIRIMModule

from utils import parse_known_args, get_dataset, get_model, retrieve_checkpoint

from tqdm import tqdm


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
        gpus=None if args.gpus == "None" else args.gpus,
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
    test_dataset = pl_loader._make_dataset(
        "all", pl_loader.file_split if hasattr(pl_loader, "file_split") else None
    )
    # test_dataset = torch.utils.data.Subset(test_dataset, list(range(100)))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=12,
        # sampler=MC_Sampler(test_dataset, mc_samples=10),
    )
    # Create model
    model = get_model(**dict_args)

    # if not args.progress_bar:
    #     print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
    #           "check the Logger file at " + trainer.logger.log_dir + ". If you " + \
    #           "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Training
    # with torch.autograd.detect_anomaly():
    if checkpoint_path:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=model.device)["state_dict"],
            # strict=False,
        )

    out_dir = args.out_dir

    if args.dataset == "tecfideramri":
        # predict_recon(args, trainer, model, test_loader, out_dir)
        predict_uncertainty(args, trainer, model, test_loader, out_dir)
        # predict_mc(args, trainer, model, test_loader, out_dir)
    elif args.dataset == "tecfidera":
        predict_seg(args, trainer, model, test_loader, out_dir)
    else:
        raise NotImplementedError


def predict_recon(args, trainer, model, loader, out_dir):
    reconstructions = collections.defaultdict(lambda: collections.defaultdict(list))
    metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    predictions = trainer.predict(model, dataloaders=loader, return_predictions=True,)

    for loss_dict, (fname, slice_int, pred) in predictions:
        fname = fname[0]
        reconstructions[fname]["reconstruction"].append(
            (slice_int, pred[-1].squeeze().numpy())
        )
        for metric, value in loss_dict.items():
            metric_dict[fname][metric].append(value.detach().mean().numpy())

    for fname in reconstructions:
        reconstructions[fname]["reconstruction"] = np.stack(
            [out for _, out in sorted(reconstructions[fname]["reconstruction"])]
        )

    save_h5(out_dir=out_dir, data_dict=reconstructions, metric_dict=metric_dict)
    get_metrics(input_dir=out_dir)


@torch.no_grad()
def predict_recon_v2(args, trainer, model, loader, out_dir):
    reconstructions = collections.defaultdict(lambda: collections.defaultdict(list))
    metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    predictions = []
    with torch.cuda.device(args.gpus[0]):
        model.eval()
        model.cuda()

        for idx, batch in enumerate(tqdm(loader)):
            # print(torch.max(batch[0][0]))
            for k, item in enumerate(batch):
                if isinstance(item, torch.Tensor):
                    batch[k] = item.cuda()
                elif isinstance(item, (list, tuple)):
                    for j, c in enumerate(item):
                        if isinstance(c, torch.Tensor):
                            batch[k][j] = c.cuda()

            output = model.predict_step(batch, idx)
            predictions.append(output)

    for loss_dict, (fname, slice_int, pred) in predictions:
        fname = fname[0]
        # pred = (
        #     (torch.abs(pred[-1]) / torch.abs(pred[-1]).amax()).squeeze().cpu().numpy()
        # )
        pred = pred[-1].squeeze().cpu().numpy()
        reconstructions[fname]["reconstruction"].append((slice_int, pred))
        for metric, value in loss_dict.items():
            metric_dict[fname][metric].append(value.detach().mean().cpu().numpy())

    for fname in reconstructions:
        reconstructions[fname]["reconstruction"] = np.stack(
            [out for _, out in sorted(reconstructions[fname]["reconstruction"])]
        )

    save_h5(out_dir=out_dir, data_dict=reconstructions, metric_dict=metric_dict)
    get_metrics(input_dir=out_dir)


def predict_seg(args, trainer, model, loader, out_dir):
    segmentations = collections.defaultdict(lambda: collections.defaultdict(list))
    metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    predictions = trainer.predict(model, dataloaders=loader, return_predictions=True,)

    for loss_dict, (fname, slice_int), pred in predictions:
        fname = fname[0]
        segmentations[fname]["segmentation"].append((slice_int, pred.squeeze().numpy()))
        for metric, value in loss_dict.items():
            metric_dict[fname][metric].append(value.detach().mean().numpy())

    for fname in segmentations:
        segmentations[fname]["segmentation"] = np.stack(
            [out for _, out in sorted(segmentations[fname]["segmentation"])],
        )

    save_h5(out_dir=out_dir, data_dict=segmentations, metric_dict=metric_dict)
    get_metrics(input_dir=out_dir)


def save_h5(out_dir, data_dict, metric_dict):
    print(out_dir)
    for fname, recons in data_dict.items():
        with h5py.File(os.path.join(out_dir, fname), "w") as hf:
            for data_key, data in recons.items():
                hf.create_dataset(data_key, data=data)
            for metric, metric_list in metric_dict[fname].items():
                hf.attrs[metric] = np.array(metric_list)


def predict_uncertainty(args, trainer, model, loader, out_dir):
    uncertainty = collections.defaultdict(lambda: collections.defaultdict(list))
    metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    predictions = trainer.predict(model, dataloaders=loader, return_predictions=True,)
    for loss_dict, (fname, slice_int, pred) in predictions:
        fname = fname[0]
        pred_stack = torch.stack(pred, dim=0)
        pred_stack = torch.abs(pred_stack) / torch.abs(pred_stack).amax((-1, -2), True)
        un_stack = torch.sqrt(
            torch.square(pred_stack - pred_stack[-1]).sum(0) / pred_stack.shape[0]
        )
        uncertainty[fname]["uncertainty"].append(
            (slice_int, un_stack.squeeze().numpy())
        )
        for metric, value in loss_dict.items():
            metric_dict[fname][metric].append(value.numpy())

    for fname in uncertainty:
        uncertainty[fname]["uncertainty"] = np.stack(
            [out for _, out in sorted(uncertainty[fname]["uncertainty"])]
        )

    save_h5(
        out_dir=out_dir, data_dict=uncertainty, metric_dict=metric_dict,
    )


def predict_mc(args, trainer, model, loader, out_dir, mc_samples=10):
    uncertainty = collections.defaultdict(lambda: collections.defaultdict(list))
    tmp_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    with torch.cuda.device(args.gpus[0]):
        with torch.no_grad():
            model.eval()
            enable_dropout(model)
            model.cuda()

            tmp = []
            for idx, batch in enumerate(tqdm(loader)):
                for k, item in enumerate(batch):
                    if isinstance(item, torch.Tensor):
                        batch[k] = item.cuda()
                    elif isinstance(item, (list, tuple)):
                        for j, c in enumerate(item):
                            if isinstance(c, torch.Tensor):
                                batch[k][j] = c.cuda()

                prediction = model.predict_step(batch, idx)
                loss_dict, (fname, slice_int, pred) = prediction
                fname = fname[0]
                # print(idx, slice_int, fname)
                if torch.is_complex(pred[-1]):
                    pred = [torch.abs(i) for i in pred]
                tmp.append(
                    (pred[-1] / pred[-1].amax((-1, -2), True)).squeeze().cpu().numpy()
                )
                for metric, value in loss_dict.items():
                    tmp_dict[fname][metric].append(value.cpu().numpy())

                if (idx + 1) % mc_samples == 0 and idx > 0:
                    uncertainty[fname]["mc_sample_variance"].append(
                        (slice_int, np.std(np.stack(tmp, axis=0), axis=0))
                    )
                    for fname, metric_list_dict in tmp_dict.items():
                        for metric, value in metric_list_dict.items():
                            metric_dict[fname][metric].append(np.mean(value))
                            metric_dict[fname][f"{metric}_std"].append(np.std(value))
                    tmp = []
                    tmp_dict = collections.defaultdict(
                        lambda: collections.defaultdict(list)
                    )

    for fname in uncertainty:
        uncertainty[fname]["mc_sample_variance"] = np.stack(
            [out for _, out in sorted(uncertainty[fname]["mc_sample_variance"])]
        )

    save_h5(
        out_dir=out_dir, data_dict=uncertainty, metric_dict=metric_dict,
    )


def get_metrics(input_dir):
    results = {}
    files = sorted(os.listdir(input_dir))
    for f in files:
        input_path = os.path.join(input_dir, f)
        hf = h5py.File(input_path)
        hf_attr = dict(hf.attrs)

        for key, value in hf_attr.items():
            if key not in results:
                results[key] = list(value)
            else:
                results[key].extend(list(value))

    for key, value in results.items():
        print(f"{key}: \t {np.nanmean(np.ma.masked_invalid(np.array(value)))}")
    print("---------")

    return results


def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith("Dropout"):
            each_module.train()


class MC_Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, mc_samples=50) -> None:
        self.dataset = data_source
        self.mc_samples = mc_samples

    def __iter__(self):
        return iter(
            [i for i in range(len(self.dataset)) for _ in range(self.mc_samples)]
        )

    def __len__(self):
        return len(self.dataset) * self.mc_samples


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
        "--gpus", nargs="+", default=[0], type=int, help="Which gpus to use."
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
