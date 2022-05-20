from logging import exception
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
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
            fname, input, target = batch
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
            prediction = torch.nn.functional.softmax(outputs[-1], dim=1)
            prediction = torch.argmax(prediction, dim=1)

            self.inputs = input.detach().cpu()
            self.targets = target.detach().cpu()
            self.predictions = prediction.detach().cpu()
            self.metrics = outputs[0]

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
            fname, input, target = batch
            if len(input.shape) == 5:
                input = rearrange(input, "b t c h w -> (b t) c h w")
            if len(target.shape) == 5:
                target = rearrange(target, "b t c h w -> (b t) c h w")
            self.num_classes = target.shape[1]
            target = torch.argmax(target, dim=1)
            prediction = torch.nn.functional.softmax(outputs[-1], dim=1)
            prediction = torch.argmax(prediction, dim=1)

            self.inputs = input.detach().cpu()
            self.targets = target.detach().cpu()
            self.predictions = prediction.detach().cpu()
            self.metrics = outputs[0]

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


class LogSegmentationMasksTECFIDERA(pl.Callback):
    def __init__(self, num_examples=5):
        super().__init__()
        self.num_examples = num_examples
        self.class_labels = {
            4: {0: "Background", 1: "Graymatter", 2: "Whitematter", 3: "Lesion"},
            2: {0: "Background", 1: "Lesion"},
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
            if len(batch) > 3:
                input = outputs[1][-1][-1]
                target = batch
            else:
                fname, input, target = batch
            if len(input.shape) == 5:
                input = rearrange(input, "b t c h w -> (b t) c h w")
                target = rearrange(target, "b t c h w -> (b t) c h w")

            if torch.is_complex(input):
                input = torch.abs(input)
            else:
                try:
                    input = torch.abs(
                        torch.view_as_complex(
                            rearrange(
                                input, "b (c i) h w -> b c h w i", i=2
                            ).contiguous()
                        )
                    )
                except Exception:
                    pass

            self.num_classes = target.shape[1]
            target = torch.argmax(target, dim=1)
            prediction = torch.nn.functional.softmax(outputs[-1], dim=1)
            prediction = torch.argmax(prediction, dim=1)

            self.inputs = input.detach().cpu()
            self.targets = target.detach().cpu()
            self.predictions = prediction.detach().cpu()
            self.metrics = outputs[0]

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


class LogUncertaintyTECFIDERA(pl.Callback):
    def __init__(self, num_examples=5):
        super().__init__()
        self.num_examples = num_examples

        self.inputs = []
        self.targets = []
        self.uncertaity = []
        self.predictions = []
        self.metrics = []

        self.exit = False

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if len(self.inputs) < self.num_examples and not self.exit:
            if len(batch) > 3:
                input = outputs[1][-1][-1]
                target = batch
            else:
                fname, input, target = batch
            if len(input.shape) == 5:
                input = rearrange(input, "b t c h w -> (b t) c h w")
                target = rearrange(target, "b t c h w -> (b t) c h w")

            if torch.is_complex(input):
                input = torch.abs(input)
            else:
                try:
                    input = torch.abs(
                        torch.view_as_complex(
                            rearrange(
                                input, "b (c i) h w -> b c h w i", i=2
                            ).contiguous()
                        )
                    )
                except Exception:
                    pass

            self.num_classes = target.shape[1]

            if outputs[-1].shape[1] == target.shape[1]:
                print("Uncertainty Callback disabled.")
                self.exit = True

            self.inputs.append(input.detach().cpu())
            self.uncertaity.append(
                outputs[-1][:, self.num_classes :, ...].detach().cpu()
            )
            self.metrics.append(outputs[0])

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if not self.exit:
            num_examples = min(self.num_examples, len(self.inputs))
            image_list = []
            captions = []
            # print(len(self.inputs), len(self.uncertaity), len(self.metrics))
            for i, (input, uncertainty, metric) in enumerate(
                zip(self.inputs, self.uncertaity, self.metrics,)
            ):
                image = input[0, ...]
                image = image / image.amax()

                uncertainty = uncertainty[0, ...]
                uncertainty_list = [
                    (uncertainty[j, ...] / uncertainty[j, ...].amax()).unsqueeze(0)
                    for j in range(uncertainty.shape[0])
                ]

                image_grid = torchvision.utils.make_grid(
                    [image] + uncertainty_list,
                    nrow=len(uncertainty_list) + 1,
                    scale_each=True,
                    normalize=True,
                )
                image_list.append(image_grid)
                caption_str = f"Uncertainty over the different classes. DSC: {metric['dice_score'].detach().mean():.3f}"
                captions.append(caption_str)

                if i >= num_examples:
                    break

            if image_list:
                trainer.logger.log_image(
                    key="Uncertainty", images=image_list, caption=captions
                )
        self.inputs = []
        self.uncertainty = []
        self.metrics = []


class LogSegmentationMasksRECSEGTECFIDERA(pl.Callback):
    def __init__(self, num_examples=5):
        super().__init__()
        self.num_examples = num_examples
        self.class_labels = {
            4: {0: "Background", 1: "Graymatter", 2: "Whitematter", 3: "Lesion"},
            2: {0: "Background", 1: "Lesion"},
        }

        self.inputs = []
        self.targets = []
        self.predictions = []
        self.metrics = []
        self.captions = []

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        (
            y,
            sensitivity_maps,
            mask,
            init_pred,
            target,
            fname,
            slice_num,
            _,
            segmentation,
        ) = batch
        if (int(slice_num) > 70 and int(slice_num) < 120) or (
            len(self.predictions) > self.num_examples
        ):
            input = outputs[1][-1][-1]

            if torch.is_complex(input):
                input = torch.abs(input)
            else:
                try:
                    input = torch.abs(
                        torch.view_as_complex(
                            rearrange(
                                input, "b (c i) h w -> b c h w i", i=2
                            ).contiguous()
                        )
                    )
                except Exception:
                    pass

            self.num_classes = segmentation.shape[-3]
            target = torch.argmax(segmentation.squeeze(), dim=-3)
            prediction = torch.nn.functional.softmax(outputs[-1], dim=1)
            prediction = torch.argmax(prediction, dim=1)

            if len(target.shape) < 3:
                target = target.unsqueeze(0)

            self.inputs.append(input[0].detach().cpu())
            self.targets.append(target[0].detach().cpu())
            self.predictions.append(prediction[0].detach().cpu())
            self.captions.append(f"{fname[0][:-3]}_slice_{int(slice_num)}")
            self.metrics.append(f"DSC: {outputs[0]['dice_score'].detach().mean():.3f}")

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        num_examples = min(self.num_examples, len(self.predictions))

        image_list = []
        masks = []
        captions = []
        for i, (input, pred, target, cap, metric) in enumerate(
            zip(
                self.inputs,
                self.predictions,
                self.targets,
                self.captions,
                self.metrics,
            )
        ):

            image = input
            image = image / image.max() * 255
            image = image.cpu().numpy().astype(np.uint8)

            target = target.cpu().numpy().astype(np.uint8)

            prediction = pred.cpu().numpy().astype(np.uint8)

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
            captions.append(f"{cap}: {metric}")

            if i >= num_examples:
                break

        if image_list:
            trainer.logger.log_image(
                key="Segmentations", images=image_list, masks=masks, caption=captions
            )
            del image_list
            del masks
            del captions
        self.inputs = []
        self.targets = []
        self.predictions = []


class LogIntermediateReconstruction(pl.Callback):
    def __init__(self, num_examples=5) -> None:
        super().__init__()
        self.num_examples = num_examples

        self.predictions = []
        self.captions = []
        self.metrics = []
        self.nrows = []

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx,
    ) -> None:
        (
            y,
            sensitivity_maps,
            mask,
            init_pred,
            target,
            fname,
            slice_num,
            _,
            segmentation,
        ) = batch
        if (int(slice_num) > 70 and int(slice_num) < 120) or (
            len(self.predictions) > self.num_examples
        ):
            preds = outputs[1]
            metric = outputs[0]
            if isinstance(preds, list):
                self.nrows.append(len(preds[-1]))
                preds = [i[0].unsqueeze(0) for j in preds for i in j]
                if len(preds) > 1:
                    pred_stack = torch.stack(preds)
                    pred_stack = torch.abs(pred_stack) / torch.abs(pred_stack).amax(
                        (-1, -2), True
                    )
                    preds = list(pred_stack.detach().cpu())
                else:
                    preds = preds[-1].cpu()

            self.predictions.append(preds)
            self.captions.append(f"{fname[0][:-3]}_slice_{int(slice_num)}")
            self.metrics.append(
                (
                    f"psnr: {metric['psnr'].detach().mean():.3f}",
                    f"ssim: {metric['ssim'].detach().mean():.3f}",
                )
            )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        num_examples = min(self.num_examples, len(self.predictions))

        image_list = []
        captions = []
        for i, (pred, cap, metric, nrow) in enumerate(
            zip(self.predictions, self.captions, self.metrics, self.nrows)
        ):
            image_grid = torchvision.utils.make_grid(pred, nrow=nrow)
            cap_str = f"{cap}: intermediate results. {metric[0]}, {metric[1]}."

            image_list.append(image_grid)
            captions.append(cap_str)

            if i >= num_examples:
                break

        if image_list:
            trainer.logger.log_image(
                key="Intermediate results", images=image_list, caption=captions
            )
            del image_list
            del captions

        self.predictions = []
        self.captions = []
        self.metrics = []
        self.nrows = []


class LogReconstructionTECFIDERA(pl.Callback):
    def __init__(self, num_examples=5):
        super().__init__()
        self.num_examples = num_examples

        self.targets = []
        self.predictions = []
        self.captions = []
        self.std = []
        self.masks = []
        self.metrics = []

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        (
            y,
            sensitivity_maps,
            mask,
            init_pred,
            target,
            fname,
            slice_num,
            _,
            segmentation,
        ) = batch
        if (int(slice_num) > 70 and int(slice_num) < 120) or (
            len(self.predictions) > self.num_examples
        ):
            preds = outputs[1]
            metric = outputs[0]
            # print(metric)
            if isinstance(preds, list):
                preds = [i[0].unsqueeze(0) for j in preds for i in j]
                if len(preds) > 1:
                    pred_stack = torch.stack(preds, dim=0)
                    pred_stack = torch.abs(pred_stack) / torch.abs(pred_stack).amax(
                        (-1, -2), True
                    )
                    # weights = F.softmax(
                    #     torch.logspace(
                    #         -1, 0, steps=len(preds), device=pl_module.device
                    #     ),
                    #     dim=0,
                    # )
                    uncertainty = (
                        torch.sqrt(
                            torch.square(pred_stack - pred_stack[-1]).sum(0)
                            / pred_stack.shape[0]
                        )
                        .detach()
                        .cpu()
                    )
                    # uncertainty = torch.std(pred_stack, dim=0).detach()
                else:
                    uncertainty = None

                preds = preds[-1]
            else:
                uncertainty = None

            if isinstance(mask, list):
                mask = mask[0]

            self.targets.append(target[:, 0, :, :].detach().cpu())
            self.predictions.append(preds.detach().cpu())
            self.captions.append(f"{fname[0][:-3]}_slice_{int(slice_num)}")
            self.std.append(uncertainty)
            self.masks.append(mask.detach().cpu())
            self.metrics.append(
                (
                    f"psnr: {metric['psnr'].detach().mean():.3f}",
                    f"ssim: {metric['ssim'].detach().mean():.3f}",
                )
            )

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        num_examples = min(self.num_examples, len(self.predictions))

        image_list = []
        captions = []
        for i, (target, pred, cap, uncertainty, mask, metric) in enumerate(
            zip(
                self.targets,
                self.predictions,
                self.captions,
                self.std,
                self.masks,
                self.metrics,
            )
        ):
            target = torch.abs(target) / torch.abs(target).amax()
            pred = torch.abs(pred) / torch.abs(pred).amax()
            error = torch.abs(target - pred)
            if uncertainty is not None:
                uncertainty = uncertainty / uncertainty.amax()
            else:
                uncertainty = torch.zeros_like(pred)

            image_grid = torchvision.utils.make_grid(
                [pred, target, error, uncertainty, mask.squeeze().unsqueeze(0)], nrow=5
            )
            cap_str = f"{cap}: reconstruction, target, error, uncertainty, mask. {metric[0]}, {metric[1]}."

            image_list.append(image_grid)
            captions.append(cap_str)

            if i >= num_examples:
                break

        if image_list:
            trainer.logger.log_image(
                key="Reconstructions", images=image_list, caption=captions
            )
            del image_list
            del captions

        self.targets = []
        self.predictions = []
        self.captions = []
        self.std = []
        self.masks = []
        self.metrics = []


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
    def on_fit_start(self, trainer, pl_module):
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
    def on_fit_start(self, trainer, pl_module):
        if pl_module.training:
            pl_module.eval()

        dummy_input = pl_module.example_input_array
        if not isinstance(dummy_input, list):
            dummy_input = [dummy_input]
        dummy_input = [i.to(pl_module.device) for i in dummy_input]

        starter, ender = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        repetitions = 200
        timings = torch.zeros(repetitions)

        for _ in range(25):
            _ = pl_module(*dummy_input)

        for rep in range(repetitions):
            starter.record()
            _ = pl_module(*dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        # timings /= np.prod(dummy_input[0].shape[:1])

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
