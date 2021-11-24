import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, LearningRateMonitor
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from einops import rearrange

from dataloader import DataModule
from pl_model import UnetModule


class LogCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name, params, trainer.current_epoch)

class LogSegmentationMasksSKMTEA(pl.Callback):
    def __init__(self, num_examples=5):
        super().__init__()
        self.num_examples = num_examples
        self.class_labels = {
            1: "Patellar Cartilage",
            2: "Femoral Cartilage",
            3: "Tibial Cartilage - Medial",
            4: "Tibial Cartilage - Lateral",
            5: "Meniscus - Medial",
            6: "Meniscus - Lateral"
        }
        self.inputs = []
        self.targets = []
        self.predictions = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            input, target = batch
            input = torch.abs(torch.view_as_complex(rearrange(input, "b (c i) h w -> b c h w i", i=2)))
            target = torch.argmax(target, dim=1)
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            self.inputs = input
            self.targets = target
            self.predictions = prediction


    def on_validation_end(self, trainer, pl_module):
        num_examples = min(self.num_examples, self.inputs.shape[0])
        image_list = []
        masks = []
        for i in range(num_examples):
            image = self.inputs[i, :, :, :]
            image = image/image.max()*255
            image = rearrange(image, "c h w -> h w c")
            image = image.cpu().numpy().astype(np.int8)

            target = self.targets[i, :, :, :]
            target = rearrange(target, "c h w -> h w c").cpu().numpy().astype(np.int8)

            prediction = self.predictions[i, :, :, :]
            prediction = rearrange(target, "c h w -> h w c").cpu().numpy().astype(np.int8)

            image_list.append(image)
            mask_dict = {
                "predictions":{
                    "mask_data": prediction,
                    "class_labels": self.class_labels
                },
                "groud_truth":{
                    "mask_data": target,
                    "class_labels": self.class_labels
                }
            }
            masks.append(mask_dict)

        trainer.logger.log_image("Predictions", image_list, masks)

class PrintCallback(pl.Callback):
    def __init__(self):
        super().__init__()
            
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} finished.")

def train(args):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    pl.seed_everything(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create a PyTorch Lightning trainer
    callbacks = []
    modelcheckpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1,
                                      save_last=True, filename='{epoch}-{val_loss:.4f}')
    callbacks.append(modelcheckpoint)
    callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=5))
    callbacks.append(ModelSummary(max_depth=1))
    callbacks.append(TQDMProgressBar(refresh_rate=1 if args.progress_bar else 0))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    if args.wandb:
        wandb_logger = WandbLogger(project="mri-segmentation", log_model="all", entity="lysander")
    else:
        callbacks.append(LogCallback())
    if not args.progress_bar:
        callbacks.append(PrintCallback())
        
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         auto_select_gpus=False,
                         gpus=[0],#None if args.gpus == "None" else int(args.gpus),
                         max_epochs=args.epochs,
                         callbacks=callbacks,
                         auto_scale_batch_size='binsearch' if args.auto_batch else None,
                         auto_lr_find=True if args.auto_lr else False,
                         precision=args.precision,
                         limit_train_batches=args.train_limit,
                         limit_val_batches=args.val_limit,
                         limit_test_batches=args.test_limit,
                         accumulate_grad_batches=args.grad_batch,
                         resume_from_checkpoint=args.checkpoint_path,
                         gradient_clip_val=args.grad_clip,
                         benchmark=True if args.benchmark else False,
                         plugins=args.plugins,
                         profiler=args.profiler if args.profiler else None,
                         enable_model_summary = False,
                         logger=wandb_logger if args.wandb else True,
                         fast_dev_run=True if args.fast_dev_run else False)
    trainer.logger._default_hp_metric = None
    trainer.logger._log_graph = False
    
    dict_args = vars(args)
    # Create dataloaders
    pl_loader = DataModule(**dict_args)
    # Create model
    model = UnetModule(**dict_args)
        
    if not args.progress_bar:
        print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
              "check the TensorBoard file at " + trainer.logger.log_dir + ". If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Training
    # with torch.autograd.detect_anomaly():
    trainer.tune(model, pl_loader)
    if args.wandb:
        wandb_logger.watch(model, log="all")
    trainer.fit(model, pl_loader)
    if args.wandb:
        wandb_logger.unwatch(model)
    print(modelcheckpoint.best_model_path)


if __name__ == '__main__':
    print("Check the Tensorboard to monitor training progress.")
    parser = argparse.ArgumentParser()
    
    # dataset hyperparameters
    parser = DataModule.add_data_specific_args(parser)

    # Model hyperparameters
    parser = UnetModule.add_model_specific_args(parser)
    
    # trainer hyperparameters
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of epochs to train.')
    
    parser.add_argument('--precision', default=32, type=int,
                        choices=[16, 32, 64, 'bf16'],
                        help='At what precision the model should train.')
    parser.add_argument('--grad_batch', default=1, type=int,
                        help='Accumulate gradient to simulate larger batch sizes.')
    parser.add_argument('--grad_clip', default=None, type=float,
                        help='Clip the gradient norm.')
    parser.add_argument('--plugins', default=None, type=str,
                        help='Modify the multi-gpu training path. See docs lightning docs for details.')
    
    parser.add_argument('--gpus', default=1, type=str,
                        help='Which gpus to use.')
    
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Continue training from this checkpoint.')
    
    # other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--log_dir', default='logs/', type=str,
                        help='Directory where the PyTorch Lightning logs ' + \
                             'should be created.')
    
    parser.add_argument('--train_limit', default=1.0, type=float,
                        help='Percentage of data to train on.')
    parser.add_argument('--val_limit', default=1.0, type=float,
                        help='Percentage of data to validate with.')
    parser.add_argument('--test_limit', default=1.0, type=float,
                        help='Percentage of data to test with.')
    
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation. '+ \
                             'Not to be used in conjuction with SLURM jobs.')
    parser.add_argument('--auto_lr', action='store_true',
                        help='When used tries to automatically set an appropriate learning rate.')
    parser.add_argument('--auto_batch', action='store_true',
                        help='When used tries to automatically set an appropriate batch size.')
    parser.add_argument('--benchmark', action='store_true',
                        help='Enables cudnn auto-tuner.')

    parser.add_argument('--wandb', action='store_true',
                        help='Enables logging to wandb, otherwise uses tensorbaord.')

    parser.add_argument('--fast_dev_run', action='store_true',
                        help='Runs a single batch for train, val and test.')
    
    parser.add_argument('--profiler', default=None, type=str,
                        choices=['simple', 'advanced', 'pytorch'],
                        help='Code profiler.')
    
    args = parser.parse_args()
    train(args)