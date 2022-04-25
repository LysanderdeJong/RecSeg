import os
import json
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from datasets import MRISliceDataset, TecFidera, brain_dwi, skmtea


class SKMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        mri_data_path,
        segmentation_path,
        annotation_path=None,
        seq_len=1,
        use_cache=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.file_split = {}

        if annotation_path is None:
            file_names = os.listdir(os.path.join(data_root, mri_data_path))
            file_names = [i[:-3] for i in file_names]
            self.file_split["all"] = file_names
        else:
            data_splits = os.listdir(os.path.join(data_root, annotation_path))
            data_splits = [i[:-5] for i in data_splits]
            for split in data_splits:
                with open(
                    os.path.join(data_root, annotation_path, f"{split}.json")
                ) as f:
                    config = json.load(f)
                    self.file_split[split] = [
                        i["file_name"][:-3] for i in config["images"]
                    ]

    def _make_dataset(self, split, data_files):
        return skmtea(
            split,
            data_files,
            self.hparams.data_root,
            self.hparams.mri_data_path,
            self.hparams.segmentation_path,
            self.hparams.seq_len,
            self.hparams.compact_masks,
            self.hparams.use_cache,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.hparams.annotation_path:
                self.train = self._make_dataset("train", self.file_split)
                self.val = self._make_dataset("val", self.file_split)
            else:
                dataset = self._make_dataset("all", self.file_split)
                self.train, self.val = random_split(
                    dataset,
                    [
                        int(self.hparams.train_fraction * len(dataset)),
                        len(dataset)
                        - (int(self.hparams.train_fraction * len(dataset))),
                    ],
                )
                self.val, self.test = random_split(
                    self.val, [len(self.val) // 2, len(self.val) - len(self.val) // 2]
                )

        if (stage == "test" or stage is None) and self.hparams.annotation_path:
            self.test = self._make_dataset("test", self.file_split)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument(
            "--data_root",
            default="/data/projects/recon/data/public/qdess/v1-release/",
            type=str,
            help="Path to the data root",
        )
        parser.add_argument(
            "--mri_data_path",
            default="files_recon_calib-24/",
            type=str,
            help="Path to the raw mri data from the root.",
        )
        parser.add_argument(
            "--segmentation_path",
            default="segmentation_masks/raw-data-track/",
            type=str,
            help="Path to the segmentation maps from the root.",
        )
        parser.add_argument(
            "--annotation_path",
            default="annotations/v1.0.0/",
            type=str,
            help="Path to the annotation maps from the root. Optional",
        )
        parser.add_argument(
            "--seq_len", default=1, type=int, help="Size of the slice looked at."
        )
        parser.add_argument(
            "--train_fraction",
            default=0.85,
            type=float,
            help="Fraction of th data used for training.",
        )
        parser.add_argument(
            "--use_cache",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--compact_masks",
            default=False,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parent_parser


class BrainDWIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        mri_data_path,
        segmentation_path,
        seq_len=1,
        use_cache=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.file_split = {}

        file_names = os.listdir(os.path.join(data_root, segmentation_path))
        file_names = [i[:-11] for i in file_names]
        self.file_split["all"] = file_names

    def _make_dataset(self, split, data_files):
        return brain_dwi(
            split,
            data_files,
            self.hparams.data_root,
            self.hparams.mri_data_path,
            self.hparams.segmentation_path,
            self.hparams.seq_len,
            self.hparams.use_cache,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = self._make_dataset("all", self.file_split)
            self.train, self.val = random_split(
                dataset,
                [
                    int(self.hparams.train_fraction * len(dataset)),
                    len(dataset) - (int(self.hparams.train_fraction * len(dataset))),
                ],
            )
            self.val, self.test = random_split(
                self.val, [len(self.val) // 2, len(self.val) - len(self.val) // 2]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument(
            "--data_root",
            default="/data/projects/dwi_aisd/",
            type=str,
            help="Path to the data root",
        )
        parser.add_argument(
            "--mri_data_path",
            default="DWIs_nii/",
            type=str,
            help="Path to the raw mri data from the root.",
        )
        parser.add_argument(
            "--segmentation_path",
            default="masks_DWI/",
            type=str,
            help="Path to the segmentation maps from the root.",
        )
        parser.add_argument(
            "--annotation_path",
            default=None,
            type=str,
            help="Path to the annotation maps from the root. Optional",
        )
        parser.add_argument(
            "--seq_len", default=1, type=int, help="Size of the slice looked at."
        )
        parser.add_argument(
            "--train_fraction",
            default=0.85,
            type=float,
            help="Fraction of th data used for training.",
        )
        parser.add_argument(
            "--use_cache",
            default=False,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--compact_masks",
            default=False,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parent_parser


class TecFideraDataModule(pl.LightningDataModule):
    def __init__(
        self, data_root, seq_len=1, use_cache=True, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.file_split = {}

        file_names = os.listdir(os.path.join(data_root))
        self.file_split["all"] = file_names

    def _make_dataset(self, split, data_files):
        return TecFidera(
            split,
            data_files,
            self.hparams.data_root,
            self.hparams.seq_len,
            self.hparams.use_cache,
            self.hparams.compact_masks,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = self._make_dataset("all", self.file_split)
            self.train, self.val = random_split(
                dataset,
                [
                    int(self.hparams.train_fraction * len(dataset)),
                    len(dataset) - (int(self.hparams.train_fraction * len(dataset))),
                ],
            )
            self.val, self.test = random_split(
                self.val, [len(self.val) // 2, len(self.val) - len(self.val) // 2]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument(
            "--data_root",
            default="/data/projects/tecfidera/data/h5_recon_dataset/",
            type=str,
            help="Path to the data root",
        )
        parser.add_argument(
            "--seq_len", default=1, type=int, help="Size of the slice looked at."
        )
        parser.add_argument(
            "--train_fraction",
            default=0.85,
            type=float,
            help="Fraction of th data used for training.",
        )
        parser.add_argument(
            "--use_cache",
            default=False,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--compact_masks",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parent_parser


class TecFideraMRIDataModule(pl.LightningDataModule):
    def __init__(
        self, data_root, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _make_dataset(self):
        return MRISliceDataset(
            root=self.hparams.data_root,
            challenge=self.hparams.challenge,
            sample_rate=self.hparams.sample_rate,
            mask_type=self.hparams.mask_type,
            shift_mask=self.hparams.shift_mask,
            accelerations=self.hparams.accelerations,
            center_fractions=self.hparams.center_fractions,
            scale=self.hparams.mask_center_scale,
            normalize_inputs=self.hparams.normalize_inputs,
            crop_size=self.hparams.crop_size,
            crop_before_masking=self.hparams.crop_before_masking,
            kspace_zero_filling_size=self.hparams.kspace_zero_filling_size,
            fft_type=self.hparams.fft_type_data,
            use_seed=self.hparams.use_seed,
            segmentation=self.hparams.segmentation,
            seq_len=self.hparams.seq_len,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = self._make_dataset()
            self.train, self.val = random_split(
                dataset,
                [
                    int(self.hparams.train_fraction * len(dataset)),
                    len(dataset) - (int(self.hparams.train_fraction * len(dataset))),
                ],
            )
            self.val, self.test = random_split(
                self.val, [len(self.val) // 2, len(self.val) - len(self.val) // 2]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument(
            "--data_root",
            default="/data/projects/tecfidera/data/h5_recon_dataset/",
            type=str,
            help="Path to the data root.",
        )
        parser.add_argument(
            "--challenge",
            default="multicoil",
            type=str,
            help="Challange category taken from FastMRI",
        )
        parser.add_argument(
            "--sample_rate",
            default=1.0,
            type=float,
            help="Fraction of the data to use.",
        )
        parser.add_argument(
            "--mask_type",
            default="gaussian2d",
            type=str,
            help="The string representation of the mask type.",
        )
        parser.add_argument(
            "--shift_mask",
            action="store_true",
            default=False,
            help="The string representation of the mask type.",
        )
        parser.add_argument(
            "--accelerations",
            nargs="+",
            type=int,
            default=[4, 6, 8, 10],
            help="Mask acceleration factors",
        )
        parser.add_argument(
            "--center_fractions",
            nargs="+",
            type=float,
            default=[0.7, 0.7, 0.7, 0.7],
            help="Center fraction of the mask",
        )
        parser.add_argument(
            "--mask_center_scale", type=float, default=0.02, help="Mask center scale",
        )
        parser.add_argument(
            "--normalize_inputs",
            action="store_false",
            default=True,
            help="Whether or not to normaliz the input to the network",
        )
        parser.add_argument(
            "--crop_size",
            nargs="+",
            type=int,
            default=None,
            help="Size to crop kspace to",
        )
        parser.add_argument(
            "--crop_before_masking",
            action="store_false",
            default=True,
            help="Whether or not the cropping happens before or after the masking",
        )
        parser.add_argument(
            "--kspace_zero_filling_size",
            nargs="+",
            type=int,
            default=None,
            help="Size to kspace to be filled",
        )
        parser.add_argument(
            "--fft_type_data",
            default="backward",
            type=str,
            help="The type of fft normilization to use.",
        )
        parser.add_argument(
            "--use_seed",
            action="store_false",
            default=True,
            help="Whether to use the seed",
        )

        parser.add_argument(
            "--segmentation",
            action="store_true",
            default=False,
            help="Whether to use the seed",
        )

        parser.add_argument(
            "--seq_len", default=1, type=int, help="Number of slices to read at once.",
        )

        parser.add_argument(
            "--train_fraction",
            default=0.85,
            type=float,
            help="Fraction of th data used for training.",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parent_parser
