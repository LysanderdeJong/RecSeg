# RecSeg
[![CodeQL](https://github.com/LysanderdeJong/RecSeg/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/LysanderdeJong/RecSeg/actions/workflows/codeql-analysis.yml)
[![Black](https://github.com/LysanderdeJong/RecSeg/actions/workflows/black.yml/badge.svg)](https://github.com/LysanderdeJong/RecSeg/actions/workflows/black.yml)

This repository contain the code for my Master Thesis: Simultaneous Reconstruction and Segmentation of Multiple Sclerosis Brain Lesions

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li>
      <a href="#code-structure">Code Structure</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#training">Training</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
      </ul>
    </li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project


### Built With
This project is based on the following deep learning frameworks:
* [Pytorch 1.11](https://pytorch.org/docs/1.11/)
* [Pytorch Lightning 1.6.5](https://pytorch-lightning.readthedocs.io/en/1.6.5/)
* [MRIDC](https://github.com/LysanderdeJong/mridc)
* [MONAI](https://docs.monai.io/en/stable/api.html)

<!-- GETTING STARTED -->
## Getting Started

Clone this github repo using:
```sh
git clone https://github.com/LysanderdeJong/RecSeg
cd RecSeg
```

Install the provided envoriment.yml using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) as follows:
```sh
conda env create -f environment.yml
```
Activate the envorment using:
```sh
conda activate thesis
```
You should now have a working enviroment.

<!-- Code Structure -->
## Code Structure

    RecSeg
    ├───model
    │	├───attunet.py      # Contains the pytorch code and lighting module for the Attention U-net model.
    │	├───cirim.py        # Contains the pytorch code and lighting module for the CIRIM mri reconstruction model.
    │	├───idslr.py        # Contains the pytorch code and lighting module for the IDSLR srs model.
    │	├───lambda_layer.py # Contains the pytorch code for the lambda+ layer.
    │	├───pics.py         # Contains the pytorch code and lighting module for the PI Compressed Sensing.
    │	├───unet.py         # Contains the pytorch code and lighting module for the Unet segmentation model.
    │	├───unet3d.py       # Contains the pytorch code and lighting module for the #D Unet segmentation model.
    │	├───unetrecon.py    # Contains the pytorch code and lighting module for the Unet recontruction model.
    │	├───vnet.py         # Contains the pytorch code and lighting module for the Volumetric Unet segmentation model.
    │	└───zf.py           # Contains the pytorch code and lighting module to perform zero filled mri reconstruction.
    ├───callbacks.py      # Contain Pytorch Lighting callbacks responable for logging during training to WandB.
    ├───dataloader.py     # Contains code to create Lightning Data Modules for the SKM-TEA, Brain DWI and proprietairy 3D FLAIR brain MS dataset.
    ├───datasets.py       # PyTorch Datasets used to read from disk.
    ├───environment.yml   # Contain an conda enviroment used to write this projects.	
    ├───evaluate.py       # Used to evaluate trained model. Results are saved along with performance metrics.
    ├───h5_to_nii.py      # Used to convert results from evaluate.py to nifti files.
    ├───losses.py         # Contains Dice loss, Hausdorf Distance and Average Surface Distance.
    ├───pl_model.py       # Combines the CIRIM and LambdaUnet into the RecSeg Lightning module.
    ├───plots.ipynb       # Contains the code to generate comparisions plots.
    ├───predict.py        # Use a trained model to perform prediction on new data.
    ├───train.py          # Contains the code to train various models from scratch.
    └───utils.py          # Contains fucntions to selects to specified model and dataset.

<!-- USAGE EXAMPLES -->
## Usage

### Training
Training a model from scratch is as simple as excuting:
```sh
python train.py --model "model_name" --dataset "dataset_name" --gpus 0
```
You can choose from the following model aliases: 'unet', 'unetrecon', 'lambdaunet', 'vnet', 'deeplab', 'unet3d', 'attunet', 'cirim', 'recseg', 'idslr', 'pics' and 'zf'.
You can choose from the following dataset aliases: 'skmtea', 'skmteamri', 'braindwi', 'tecfidera', 'tecfideramri'
The models can be trained on gpu(s) by specifying --gpus [list of gpu ids].

You can always type:
```sh
python train.py --help
```

Examples:
```sh
python train.py --model unet --dataset braindwi --progress_bar --batch_size 8 --num_workers 16 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --chans 22 --out_chans 2 --gpus 1 --train_metric_only true
```
```sh
python train.py --model unet --dataset braindwi --progress_bar --batch_size 8 --num_workers 16 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --project dwi-segmentation --gpus 1 --train_metric_only true
```
```sh
python train.py --model attunet --dataset braindwi --progress_bar --batch_size 8 --num_workers 16 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --gpus 1 --train_metric_only true
```
```sh
python train.py --model lambdaunet --dataset braindwi --progress_bar --batch_size 8 --num_workers 8 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --gpus 1 --seq_len 1 --tr 1 --num_slices 1 --train_metric_only true
```
```sh
python train.py --model lambdaunet --dataset braindwi --progress_bar --batch_size 1 --num_workers 4 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --gpus 0 --seq_len 3 --tr 3 --num_slices 3 --train_metric_only true
```

### Evaluation
A model may be evaluated by excuting:
```sh
python evaluate.py --model "model_name" --dataset "dataset_name" --gpus 0 --eval_mode "eval_mode" --checkpoint_path "path_to_model_checkpoint"
```
Eval mode can be set to either 'segmentation', 'recon' or 'srs'.

Examples:
```sh
python evaluate.py --model unet --dataset braindwi --progress_bar --batch_size 1 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --chans 22 --project dwi-segmentation --gpus 1 --model_id 2l91xy97 --eval_mode segmentation --out_dir /data/projects/tecfidera/data/results/segmentation/eval_on_dwi/unet_3_7/h5/ --train_metric_only false
```
```sh
python evaluate.py --model unet --dataset braindwi --progress_bar --batch_size 1 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --project dwi-segmentation --gpus 1 --model_id 2u8cjt30 --eval_mode segmentation --out_dir /data/projects/tecfidera/data/results/segmentation/eval_on_dwi/unet_7_8/h5/ --train_metric_only false
```
```sh
python evaluate.py --model attunet --dataset braindwi --progress_bar --batch_size 1 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --project dwi-segmentation --gpus 1 --model_id 1kpu6ood --eval_mode segmentation --out_dir /data/projects/tecfidera/data/results/segmentation/eval_on_dwi/attunet_8_3/h5/ --train_metric_only false
```
```sh
python evaluate.py --model unet --dataset braindwi --progress_bar --batch_size 1 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --chans 22 --project dwi-segmentation --gpus 1 --model_id 2l91xy97 --eval_mode segmentation --out_dir /data/projects/tecfidera/data/results/segmentation/eval_on_dwi/unet_3_7/h5/ --train_metric_only false
```
```sh
python evaluate.py --model lambdaunet --dataset braindwi --progress_bar --batch_size 1 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --project dwi-segmentation --gpus 1 --model_id 3tii7pvo --eval_mode segmentation --out_dir /data/projects/tecfidera/data/results/segmentation/eval_on_dwi/lambdaunet_2d_3_6/h5/ --train_metric_only false --tr 1 --num_slices 1
```
```sh
python evaluate.py --model lambdaunet --dataset braindwi --progress_bar --batch_size 1 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --project dwi-segmentation --gpus 1 --model_id 3b4wv7to --eval_mode segmentation --out_dir /data/projects/tecfidera/data/results/segmentation/eval_on_dwi/lambdaunet_3_6/h5/ --train_metric_only false --tr 3 --num_slices 3 --seq_len 3
```
