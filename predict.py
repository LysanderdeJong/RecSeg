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

    dataset = pl_loader._make_dataset(
        "all", pl_loader.file_split if hasattr(pl_loader, "file_split") else None
    )
    if args.dataset == "tecfideramri":
        dataset = pl_loader._make_dataset(data_files=pl_loader.data_split["test"])
    else:
        dataset = pl_loader._make_dataset(
            "all", pl_loader.file_split if hasattr(pl_loader, "file_split") else None
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
            strict=False,
        )
    evalutaion_pipeline = Evaluate(args, trainer, model, dataset)
    evalutaion_pipeline.predict_loop()
    print(args.out_dir)
    return evalutaion_pipeline.get_metrics()


class Evaluate:
    def __init__(self, args, trainer, model, dataset):
        self.args = args
        self.trainer = trainer
        self.model = model

        dataset = torch.utils.data.Subset(
            dataset, list(range(int(len(dataset) * args.test_limit)))
        )
        self.loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            num_workers=12,
            sampler=MC_Sequential_Sampler(dataset, mc_samples=args.mc_samples),
        )

    def predict_loop(self):
        self.data_dict = defaultdict(lambda: defaultdict(list))
        self.tmp_metric_dict = defaultdict(lambda: defaultdict(list))
        self.metric_dict = defaultdict(lambda: defaultdict(list))
        device = torch.device(f"cuda:{self.args.gpus[0]}") if self.args.gpus else "cpu"
        # with torch.cuda.device(self.args.gpus[0]):
        with torch.no_grad():
            self.model.eval()
            if self.args.mc_samples > 1:
                self.enable_dropout()
            self.model.to(device)

            self.tmp_data_dict = defaultdict(lambda: defaultdict(list))
            prev_fname = "DMF008_T2_AXFLAIR_transverse.h5"
            for idx, batch in enumerate(tqdm(self.loader)):
                batch = self.move_to_gpu(batch, device)
                if (idx + 1) % self.args.mc_samples:
                    self.disable_dropout()

                if self.args.eval_mode == "recon":
                    fname, slice_int = self.proces_recon(batch, idx)
                elif self.args.eval_mode == "segmentation":
                    fname, slice_int = self.proces_segmentation(batch, idx)
                elif self.args.eval_mode == "srs":
                    fname, slice_int = self.proces_srs(batch, idx)
                else:
                    raise NotImplementedError

                if (idx + 1) % self.args.mc_samples == 0:
                    for fname, tmp_data in self.tmp_data_dict.items():
                        for data_tag, data_list in tmp_data.items():
                            if len(data_list) > 1:
                                self.data_dict[fname][data_tag].append(
                                    (
                                        slice_int,
                                        np.std(np.stack(data_list, axis=0), axis=0),
                                    )
                                )
                            else:
                                self.data_dict[fname][data_tag].append(
                                    (slice_int, np.stack(data_list, axis=0))
                                )

                    for fname, metric_list_dict in self.tmp_metric_dict.items():
                        for metric, value in metric_list_dict.items():
                            self.metric_dict[fname][metric].append(
                                np.mean(np.ma.masked_invalid(value))
                            )
                            if self.args.mc_samples > 1:
                                self.metric_dict[fname][f"{metric}_std"].append(
                                    np.std(np.ma.masked_invalid(value))
                                )

                    self.tmp_data_dict = defaultdict(lambda: defaultdict(list))
                    self.tmp_metric_dict = defaultdict(lambda: defaultdict(list))
                    self.enable_dropout()

                if prev_fname != fname:
                    for data_label in self.data_dict[prev_fname].keys():
                        self.data_dict[prev_fname][data_label] = np.stack(
                            [
                                np.squeeze(out)
                                for _, out in sorted(
                                    self.data_dict[prev_fname][data_label]
                                )
                            ]
                        )
                    self.save_h5(
                        data_dict=self.data_dict,
                        metric_dict=self.metric_dict,
                        key=prev_fname,
                    )
                    del self.data_dict[prev_fname]
                    del self.metric_dict[prev_fname]
                prev_fname = fname
        for fname, data_label in self.data_dict.items():
            for data_label in self.data_dict[fname].keys():
                self.data_dict[fname][data_label] = np.stack(
                    [
                        np.squeeze(out)
                        for _, out in sorted(self.data_dict[fname][data_label])
                    ]
                )
        self.save_h5(data_dict=self.data_dict, metric_dict=self.metric_dict)

    def proces_recon(self, batch, idx):
        prediction = self.model.predict_step(batch, idx)
        loss_dict, (fname, slice_int, pred) = prediction
        fname = fname[0]
        # print(pred.shape)
        if torch.is_complex(pred[-1]):
            pred = [torch.abs(i) for i in pred]

        if self.args.mc_samples > 1:
            uncertainty = pred[-1].clone()
            self.tmp_data_dict[fname]["aleatoric_uncertainty"].append(
                (uncertainty / uncertainty.amax((-1, -2), True)).squeeze().cpu().numpy()
            )

        if (idx + 1) % self.args.mc_samples == 0:
            recon = pred[-1].clone()
            self.tmp_data_dict[fname]["reconstruction"].append(
                recon.squeeze().cpu().numpy()
            )

            gt = batch[4].clone()
            self.tmp_data_dict[fname]["ground_truth"].append(gt.squeeze().cpu().numpy())

            recon = pred[-1].clone()
            recon = (recon / recon.amax((-1, -2), True)).squeeze()
            gt = batch[4].clone()
            gt = (gt / gt.amax((-1, -2), True)).squeeze()
            self.tmp_data_dict[fname]["difference_error"].append(
                torch.abs(recon - gt.cpu()).cpu().numpy()
            )

            if len(pred) > 1:
                if len(pred) <= 8:
                    pred_stack = torch.stack(pred, dim=0)
                else:
                    pred_stack = torch.stack(pred[8:], dim=0)
                pred_stack = torch.abs(pred_stack) / torch.abs(pred_stack).amax(
                    (-1, -2), True
                )
                un_stack = torch.sqrt(
                    torch.square(pred_stack - pred_stack[-1]).sum(0)
                    / pred_stack.shape[0]
                )
                self.tmp_data_dict[fname]["intermediate_std"].append(
                    un_stack.squeeze().cpu().numpy()
                )

        for metric, value in loss_dict.items():
            self.tmp_metric_dict[fname][metric].append(value.cpu().numpy())
        return fname, slice_int

    def proces_segmentation(self, batch, idx):
        prediction = self.model.predict_step(batch, idx)
        loss_dict, (fname, slice_int), pred = prediction
        fname = fname[0]

        if self.args.mc_samples > 1:
            self.tmp_data_dict[fname]["epistemic_uncertainty"].append(
                torch.softmax(pred, dim=1).squeeze().cpu().numpy()
            )

        if (idx + 1) % self.args.mc_samples == 0:
            self.tmp_data_dict[fname]["segmentation"].append(
                pred.squeeze().cpu().numpy()
            )

            self.tmp_data_dict[fname]["ground_truth"].append(
                batch[-1].squeeze().cpu().numpy()
            )

            self.tmp_data_dict[fname]["difference_error"].append(batch[-1].cpu() - pred)

        for metric, value in loss_dict.items():
            self.tmp_metric_dict[fname][metric].append(value.cpu().numpy())
        return fname, slice_int

    def proces_srs(self, batch, idx):
        prediction = self.model.predict_step(batch, idx)
        loss_dict, (fname, slice_int), pred_recon, pred_seg = prediction
        fname = fname[0]
        if torch.is_complex(pred_recon[-1]):
            pred_recon = [torch.abs(i) for i in pred_recon]

        if self.args.mc_samples > 1:
            uncertainty = pred_recon[-1].clone()
            self.tmp_data_dict[fname]["aleatoric_uncertainty"].append(
                (uncertainty / uncertainty.amax((-1, -2), True)).squeeze().cpu().numpy()
            )
            self.tmp_data_dict[fname]["epistemic_uncertainty"].append(
                torch.softmax(pred_seg, dim=1).squeeze().cpu().numpy()
            )

        if (idx + 1) % self.args.mc_samples == 0:
            self.tmp_data_dict[fname]["segmentation"].append(
                pred_seg.squeeze().cpu().numpy()
            )

            self.tmp_data_dict[fname]["ground_truth_seg"].append(
                batch[-1].squeeze().cpu().numpy()
            )

            self.tmp_data_dict[fname]["difference_error_seg"].append(
                batch[-1].cpu() - pred_seg
            )

            recon = pred_recon[-1].clone()
            self.tmp_data_dict[fname]["reconstruction"].append(
                recon.squeeze().cpu().numpy()
            )

            gt = batch[4].clone()
            self.tmp_data_dict[fname]["ground_truth_recon"].append(
                gt.squeeze().cpu().numpy()
            )

            recon = pred_recon[-1].clone()
            recon = (recon / recon.amax((-1, -2), True)).squeeze()
            gt = batch[4].clone()
            gt = (gt / gt.amax((-1, -2), True)).squeeze()
            self.tmp_data_dict[fname]["difference_error_recon"].append(
                torch.abs(recon - gt.cpu()).cpu().numpy()
            )

            if len(pred_recon) > 1:
                if len(pred_recon) <= 8:
                    pred_stack = torch.stack(pred_recon, dim=0)
                else:
                    pred_stack = torch.stack(pred_recon[8:], dim=0)
                pred_stack = torch.abs(pred_stack) / torch.abs(pred_stack).amax(
                    (-1, -2), True
                )
                un_stack = torch.sqrt(
                    torch.square(pred_stack - pred_stack[-1]).sum(0)
                    / pred_stack.shape[0]
                )
                self.tmp_data_dict[fname]["intermediate_std"].append(
                    un_stack.squeeze().cpu().numpy()
                )

        for metric, value in loss_dict.items():
            self.tmp_metric_dict[fname][metric].append(value.mean().cpu().numpy())
            # print(metric, value.cpu().numpy())
        return fname, slice_int

    def move_to_gpu(self, batch, device):
        for k, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                batch[k] = item.to(device)
            elif isinstance(item, (list, tuple)):
                for j, c in enumerate(item):
                    if isinstance(c, torch.Tensor):
                        batch[k][j] = c.to(device)
                    elif isinstance(c, (list, tuple)):
                        for l, d in enumerate(c):
                            if isinstance(d, torch.Tensor):
                                batch[k][j][l] = d.to(device)
        return batch

    def save_h5(self, data_dict, metric_dict, key=None):
        if key in data_dict.keys():
            with h5py.File(os.path.join(self.args.out_dir, key), "w") as hf:
                for data_key, data in data_dict[key].items():
                    hf.create_dataset(data_key, data=data)
                for metric, metric_list in metric_dict[key].items():
                    hf.attrs[metric] = np.array(metric_list)
        else:
            for fname, recons in data_dict.items():
                with h5py.File(os.path.join(self.args.out_dir, fname), "w") as hf:
                    for data_key, data in recons.items():
                        hf.create_dataset(data_key, data=data)
                    for metric, metric_list in metric_dict[fname].items():
                        hf.attrs[metric] = np.array(metric_list)

    def get_metrics(self):
        results = {}
        files = sorted(os.listdir(self.args.out_dir))
        for f in files:
            input_path = os.path.join(self.args.out_dir, f)
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

    def enable_dropout(self):
        for each_module in self.model.modules():
            if each_module.__class__.__name__.startswith("Dropout"):
                each_module.train()

    def disable_dropout(self):
        for each_module in self.model.modules():
            if each_module.__class__.__name__.startswith("Dropout"):
                each_module.eval()


class MC_Sequential_Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, mc_samples=1) -> None:
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
