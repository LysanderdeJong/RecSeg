from itertools import product
import torch
import numpy as np
from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import wandb


class LogCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name, params, trainer.current_epoch)


class LogSegmentationMasksSKMTEA(pl.Callback):
    def __init__(self, num_examples=5):
        super().__init__()
        self.num_examples = num_examples
        self.class_labels = {
            7: {
                0: "Background",
                1: "Patellar Cartilage",
                2: "Femoral Cartilage",
                3: "Tibial Cartilage - Medial",
                4: "Tibial Cartilage - Lateral",
                5: "Meniscus - Medial",
                6: "Meniscus - Lateral",
            },
            5: {
                0: "Background",
                1: "Patellar Cartilage",
                2: "Femoral Cartilage",
                3: "Tibial Cartilage",
                4: "Meniscus",
            },
        }

        self.inputs = []
        self.targets = []
        self.predictions = []
        self.metrics = []

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx == 1:
            input, target = batch
            if len(input.shape) == 5:
                input = rearrange(input, "b t c h w -> (b t) c h w")
                target = rearrange(target, "b t c h w -> (b t) c h w")
            input = torch.abs(
                torch.view_as_complex(
                    rearrange(input, "b (c i) h w -> b c h w i", i=2).contiguous()
                )
            )
            self.num_classes = target.shape[1]
            target = torch.argmax(target, dim=1)
            prediction = torch.nn.functional.softmax(outputs[0], dim=1)
            prediction = torch.argmax(prediction, dim=1)

            self.inputs = input
            self.targets = target
            self.predictions = prediction
            self.metrics = outputs[1]

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        num_examples = min(self.num_examples, self.inputs.shape[0])
        image_list = []
        masks = []
        captions = []
        for i in range(num_examples):
            image = self.inputs[i, :, :, :]
            image = image / image.max() * 255
            image = rearrange(image, "c h w -> h w c")
            image = image.cpu().numpy().astype(np.uint8)

            target = self.targets[i, :, :].cpu().numpy().astype(np.uint8)

            prediction = self.predictions[i, :, :].cpu().numpy().astype(np.uint8)

            image_list.append(image)
            mask_dict = {
                "predictions": {
                    "mask_data": prediction,
                    "class_labels": self.class_labels[self.num_classes],
                },
                "groud_truth": {
                    "mask_data": target,
                    "class_labels": self.class_labels[self.num_classes],
                },
            }
            masks.append(mask_dict)
            caption_str = f"DSC: {self.metrics['dice_score'][i].item():.3f}"
            captions.append(caption_str)

        trainer.logger.log_image(
            key="Predictions", images=image_list, masks=masks, caption=captions
        )
        self.inputs = []
        self.targets = []
        self.predictions = []


class LogSegmentationMasksDWI(pl.Callback):
    def __init__(self, num_examples=5):
        super().__init__()
        self.num_examples = num_examples
        self.class_labels = {
            2: {0: "Background", 1: "Infarct Lesion"},
            5: {
                0: "Background",
                1: "Infarct Lesion",
                2: "Infarct Lesion",
                3: "Infarct Lesion",
                4: "MRI Infarct Lesion",
                5: "Infarct Lesion",
            },
        }

        self.inputs = []
        self.targets = []
        self.predictions = []
        self.metrics = []

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx == 1:
            input, target = batch
            if len(input.shape) == 5:
                input = rearrange(input, "b t c h w -> (b t) c h w")
            if len(target.shape) == 5:
                target = rearrange(target, "b t c h w -> (b t) c h w")
            self.num_classes = target.shape[1]
            target = torch.argmax(target, dim=1)
            prediction = torch.nn.functional.softmax(outputs[0], dim=1)
            prediction = torch.argmax(prediction, dim=1)

            self.inputs = input
            self.targets = target
            self.predictions = prediction
            self.metrics = outputs[1]

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        num_examples = min(self.num_examples, self.inputs.shape[0])
        image_list = []
        masks = []
        captions = []
        for i in range(num_examples):
            image = self.inputs[i, :, :, :]
            image -= image.min()
            image = image / image.max() * 255
            image = rearrange(image, "c h w -> h w c")
            image = image.cpu().numpy().astype(np.uint8)

            target = self.targets[i, :, :].cpu().numpy().astype(np.uint8)

            prediction = self.predictions[i, :, :].cpu().numpy().astype(np.uint8)

            image_list.append(image)
            mask_dict = {
                "predictions": {
                    "mask_data": prediction,
                    "class_labels": self.class_labels[self.num_classes],
                },
                "groud_truth": {
                    "mask_data": target,
                    "class_labels": self.class_labels[self.num_classes],
                },
            }
            masks.append(mask_dict)
            caption_str = f"DSC: {self.metrics['dice_score'][i].item():.3f}"
            captions.append(caption_str)

        trainer.logger.log_image(
            key="Predictions", images=image_list, masks=masks, caption=captions
        )
        self.inputs = []
        self.targets = []
        self.predictions = []


class PrintCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} finished.")


class NumParamCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def on_pretrain_routine_start(self, trainer, pl_module):
        trainable_params = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in pl_module.parameters())
        non_trainable_params = total_params - trainable_params
        if isinstance(
            trainer.logger.experiment, torch.utils.tensorboard.writer.SummaryWriter
        ):
            trainer.logger.log_hyperparams(
                pl_module.hparams, {"hp/total_parameters": trainable_params}
            )
            trainer.logger.log_hyperparams(
                pl_module.hparams, {"hp/trainable_parameters": trainable_params}
            )
            trainer.logger.log_hyperparams(
                pl_module.hparams, {"hp/non-trainable_parameters": non_trainable_params}
            )
        elif isinstance(trainer.logger.experiment, wandb.sdk.wandb_run.Run):
            trainer.logger.experiment.summary["total_parameters"] = total_params
            trainer.logger.experiment.summary["trainable_parameters"] = trainable_params
            trainer.logger.experiment.summary[
                "non-trainable_parameters"
            ] = non_trainable_params


class InferenceTimeCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    @torch.no_grad()
    def on_pretrain_routine_start(self, trainer, pl_module):
        if pl_module.training:
            pl_module.eval()
        dummy_input = pl_module.example_input_array
        dummy_input = dummy_input.to(pl_module.device)
        starter, ender = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        repetitions = 300
        timings = torch.zeros(repetitions)

        for _ in range(25):
            _ = pl_module(dummy_input)

        for rep in range(repetitions):
            starter.record()
            _ = pl_module(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        timings /= np.prod(
            [dummy_input.shape[i] for i in range(len(dummy_input.shape) - 2)]
        )

        print(f"Inference time: {timings.mean():0.3f}ms ± {timings.std():0.3f}ms.")

        if isinstance(
            trainer.logger.experiment, torch.utils.tensorboard.writer.SummaryWriter
        ):
            trainer.logger.log_hyperparams(
                pl_module.hparams, {"hp/inference_time": timings.mean()}
            )
        elif isinstance(trainer.logger.experiment, wandb.sdk.wandb_run.Run):
            trainer.logger.experiment.summary["inference_time"] = timings.mean()

        del dummy_input
        del timings
        pl_module.train()
