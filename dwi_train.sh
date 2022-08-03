cd /scratch/lgdejong/projects/RecSeg/
conda activate thesis

python train.py --model unet --dataset braindwi --progress_bar --precision 32 --batch_size 8 --grad_batch 4 --num_workers 16 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --chans 22 --out_chans 2 --project dwi-segmentation --gpus 1 --lr 0.02 --seed 4182830386 --wandb --train_metric_only true

python train.py --model unet --dataset braindwi --progress_bar --precision 32 --batch_size 8 --grad_batch 4 --num_workers 16 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --project dwi-segmentation --gpus 1 --lr 0.02 --seed 4182830386 --wandb --train_metric_only true

python train.py --model attunet --dataset braindwi --progress_bar --precision 32 --batch_size 8 --grad_batch 4 --num_workers 16 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --project dwi-segmentation --gpus 1 --lr 0.01 --seed 4182830386 --wandb --train_metric_only true

python train.py --model lambdaunet --dataset braindwi --progress_bar --precision 32 --batch_size 8 --grad_batch 4 --num_workers 16 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --project dwi-segmentation --gpus 1 --lr 0.01 --seed 4182830386 --wandb --seq_len 1 --tr 1 --num_slices 1 --train_metric_only true

python train.py --model lambdaunet --dataset braindwi --progress_bar --precision 32 --batch_size 1 --grad_batch 32 --num_workers 16 --data_root /data/projects/dwi_aisd/ --mri_data_path DWIs_nii/ --segmentation_path masks_DWI/ --in_chans 1 --out_chans 2 --project dwi-segmentation --gpus 0 --lr 0.01 --seed 4182830386 --wandb --seq_len 3 --tr 3 --num_slices 3 --train_metric_only true