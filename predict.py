import itertools
import os
import pickle
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Union

import hydra
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
)
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    GTSRB,
    MNIST,
    PCAM,
    STL10,
    SVHN,
    Flowers102,
    Food101,
)
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_H_14_Weights,
    ViT_L_16_Weights,
    vit_b_16,
    vit_h_14,
    vit_l_16,
)
from transformers import ViTForImageClassification

torch.set_float32_matmul_precision("highest")


def delta(ds: str):
    return {
        "stl10": 1e-3,
        "svhn": 1 / (2 * 75000),
        "pcam": 1e-5,
        "cifar10": 1e-5,
        "cifar100": 1e-5,
        "cifar10_centertrick": 1e-5,
        "cifar100_centertrick": 1e-5,
        "flowers102": 1 / (2 * 1020),
        "food101": 1 / (2 * 750 * 101),
    }[ds]


@dataclass
class TensorDataset(Dataset):
    data: Union[torch.Tensor, str, Path]
    targets: Optional[Union[torch.Tensor, str, Path]]

    def __post_init__(self):
        super().__init__()
        if isinstance(self.data, (Path, str)):
            print(f"Loading {self.data}")
            path = Path(self.data)
            if path.suffix == ".zarr":
                import dask.array as da

                self.data = torch.tensor(da.from_zarr(path).compute())
            elif path.suffix == ".pt":
                self.data = torch.load(self.data, map_location="cpu")
            else:
                raise ValueError(f"Unknown file type {path.suffix}")
        if isinstance(self.targets, (Path, str)):
            print(f"Loading {self.targets}")
            path = Path(self.targets)
            if path.suffix == ".zarr":
                import dask.array as da

                self.targets = torch.tensor(da.from_zarr(path).compute())
            elif path.suffix == ".pt":
                self.targets = torch.load(self.targets, map_location="cpu")
            else:
                raise ValueError(f"Unknown file type {path.suffix}")

    def __getitem__(self, index):
        if self.targets is None:
            return (self.data[index],)
        return (self.data[index], self.targets[index])

    def __len__(self):
        return self.data.shape[0]


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        transform=None,
        size: int = 64,
        split: Literal["train", "val"] = "train",
    ):
        self.root = Path(root)
        self.transform = transform
        self.data = []
        self.targets = []
        if split == "train":
            start_str = "train_data_batch_"
        elif split == "val":
            start_str = "val_data"
        else:
            raise ValueError(f"Unknown split {split}")
        for f in [f for f in os.listdir(self.root) if f.startswith(start_str)]:
            data = self.unpickle(self.root / f)
            self.data.append(data["data"])
            self.targets.append(data["labels"])
        self.data = np.concatenate(self.data)
        self.targets = np.concatenate(self.targets)
        self.data = self.data.reshape(-1, 3, size, size)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def unpickle(file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo)
        return dict


class StandardPCAM(PCAM):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ):
        super().__init__(root, split, transform, target_transform, download)
        targets_file = self._FILES[self._split]["targets"][0]
        with self.h5py.File(self._base_folder / targets_file) as targets_data:
            self.targets = targets_data["y"][:, 0, 0, 0]


class StandardGTSRB(GTSRB):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.targets = np.asarray(self._samples)[:, 1].astype(int)


class ProtoData(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.batch_size_physical = None

    def predict_targets(self, only_train: bool = False):
        split = self.train if only_train else self.pred
        if hasattr(split, "dataset"):
            targets = getattr(split.dataset, self.target_key)[split.indices]
        else:
            targets = getattr(split, self.target_key)
        return targets

    def predict_dataloader(
        self, only_train: bool = False, dataset: Optional[Any] = None
    ) -> EVAL_DATALOADERS:
        """Returns dataset for generating prototypes.

        Args:
            only_train (bool, optional): Only return the train split from the training dataset. Set to true for generating validation prototypes. Defaults to False.
            dataset (Optional[Any], optional): Return dataloader for a given dataset, e.g. a subset of the training data to save on computation. Defaults to None.

        Raises:
            ValueError: If only_train and dataset are both specified.

        Returns:
            EVAL_DATALOADERS: Dataloader for generating prototypes.
        """
        if not dataset:
            dataset = self.train if only_train else self.pred
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


class ProtoImageData(ProtoData):
    def __init__(
        self,
        ds: str = "MNIST",
        data_dir: str = "/raid/datasets",
        num_workers: int = 8,
        size_target: int = 224,
        center_trick: Optional[int] = None,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        flip_prob: float = 0.5,
        rand_aug_n: int = 0,
        rand_aug_m: int = 9,
        erase_prob: float = 0.0,
        use_trivial_aug: bool = False,
        n_way: int = 5,
        k_shot: int = 5,
        eval_batch_size: int = 512,
        prototypical_sampler: bool = True,
        ratio_val: float = 0.2,
        ratio_pred: float = 1.0,
        seed: int = 42,
    ):
        """_summary_

        Args:
            ds (str, optional): _description_. Defaults to "MNIST".
            data_dir (str, optional): _description_. Defaults to "/datasets".
            num_workers (int, optional): _description_. Defaults to 8.
            size_target (int, optional): _description_. Defaults to 224.
            center_trick (Optional[int], optional): _description_. Defaults to None.
            min_scale (float, optional): _description_. Defaults to 0.08.
            max_scale (float, optional): _description_. Defaults to 1.0.
            flip_prob (float, optional): _description_. Defaults to 0.5.
            rand_aug_n (int, optional): _description_. Defaults to 0.
            rand_aug_m (int, optional): _description_. Defaults to 9.
            erase_prob (float, optional): _description_. Defaults to 0.0.
            use_trivial_aug (bool, optional): _description_. Defaults to False.
            n_way (int, optional): Number of classes to sample per batch. Defaults to 5.
            k_shot (int, optional): Number of examples to sample per class in the batch. Defaults to 5.
            eval_batch_size (int, optional): _description_. Defaults to 512.
            prototypical_sampler(bool, optional): Use prototypical sampler. Defaults to True.
            ratio_va(float, optional): Ratio of validation data. Defaults to 0.2.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        self.ds = ds.upper()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.eval_batch_size = eval_batch_size
        self.target_key = "targets"  # default key
        self.collate_fn = None
        self.prototypical_sampler = prototypical_sampler
        self.ratio_val = ratio_val
        self.ratio_pred = ratio_pred
        self.seed = seed

        self.n_way = n_way
        self.k_shot = k_shot
        self.prepare_args = dict(
            download=True,
        )
        self.setup_args = dict(
            download=False,
        )
        if self.ds == "MNIST":
            mean = (0.1307,)
            std = (0.3081,)
            self.train_args = dict(train=True)
            self.test_args = dict(train=False)
            self.dataclass = MNIST
        elif self.ds == "CIFAR10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            self.train_args = dict(train=True)
            self.test_args = dict(train=False)
            self.dataclass = CIFAR10
        elif self.ds == "CIFAR100":
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
            self.train_args = dict(train=True)
            self.test_args = dict(train=False)
            self.dataclass = CIFAR100
        elif self.ds == "SVHN":
            mean = (0.4377, 0.4438, 0.4728)
            std = (0.1980, 0.2010, 0.1970)
            self.train_args = dict(split="train")
            self.test_args = dict(split="test")
            self.target_key = "labels"
            self.dataclass = SVHN
        elif self.ds == "STL10":
            mean = (0.4467, 0.4398, 0.4066)
            std = (0.2603, 0.2566, 0.2713)
            self.train_args = dict(split="train")
            self.test_args = dict(split="test")
            self.target_key = "labels"
            self.dataclass = STL10
        elif self.ds == "FLOWERS102":
            mean = (0.4302, 0.3796, 0.2946)
            std = (0.2946, 0.2465, 0.2734)
            self.train_args = dict(split="train")
            self.test_args = dict(split="test")
            self.target_key = "_labels"
            self.dataclass = Flowers102
        elif self.ds == "GTSRB":
            mean = (0.3805, 0.3484, 0.3574)
            std = (0.3031, 0.295, 0.3007)
            self.train_args = dict(split="train")
            self.test_args = dict(split="test")
            self.dataclass = StandardGTSRB
        elif self.ds == "PCAM":
            mean = (0.7008, 0.5384, 0.6916)
            std = (0.235, 0.2774, 0.2129)
            self.train_args = dict(split="train")
            self.test_args = dict(split="test")
            self.dataclass = StandardPCAM
        elif self.ds == "FOOD101":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            self.train_args = dict(split="train")
            self.test_args = dict(split="test")
            self.dataclass = Food101
            self.target_key = "_labels"
        elif self.ds == "IMAGENET64":
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
            self.train_args = dict(split="train")
            self.test_args = dict(split="val")
            self.dataclass = ImageNetDataset
            self.prepare_args = dict()
            self.setup_args = dict()
        else:
            raise ValueError(f"No implementation for dataset '{self.ds}'")
        self.inverse_transform = transforms.Normalize(
            tuple(-mu / sig for mu, sig in zip(mean, std)),
            tuple(1 / sig for sig in std),
        )
        if center_trick is not None:
            size = center_trick
        else:
            size = size_target
        transform_list = [
            transforms.Resize(size),
            transforms.CenterCrop(size_target),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        train_transform_list = [
            transforms.RandomResizedCrop(
                size,
                scale=(min_scale, max_scale),
            ),
            transforms.RandomHorizontalFlip(flip_prob),
            transforms.TrivialAugmentWide()
            if use_trivial_aug
            else transforms.RandAugment(rand_aug_n, rand_aug_m),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=erase_prob),
        ]
        if center_trick is not None:
            transform_list.insert(1, transforms.CenterCrop(size_target))
            train_transform_list.insert(1, transforms.CenterCrop(size_target))
        self.train_transform = transforms.Compose(train_transform_list)
        self.transform = transforms.Compose(transform_list)
        # Print all parameters
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")

    def prepare_data(self) -> None:
        if not hasattr(
            self, "ds_train"
        ):  # if-check to prevent repeated integrity checks with download=True
            self.dataclass(
                self.data_dir,
                **self.train_args,  # type: ignore
                **self.prepare_args,
                transform=self.train_transform,
            )
            self.dataclass(
                self.data_dir,
                **self.test_args,  # type: ignore
                **self.prepare_args,
                transform=self.transform,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.ds_train = self.dataclass(
                self.data_dir,
                **self.train_args,  # type: ignore
                **self.setup_args,
                transform=self.train_transform,
            )
            self.train, self.val = torch.utils.data.random_split(
                self.ds_train,
                [1 - self.ratio_val, self.ratio_val],
                torch.Generator().manual_seed(self.seed),
            )
            self.ds_pred = self.dataclass(
                self.data_dir,
                **self.train_args,  # type: ignore
                **self.setup_args,
                transform=self.transform,
            )
            pred_indices = subset(
                torch.tensor(getattr(self.ds_pred, self.target_key)),
                self.ratio_pred,
                balanced=True,
                seed=self.seed,
            )
            self.pred = torch.utils.data.Subset(self.ds_pred, pred_indices)
        if stage == "test" or stage is None:
            self.ds_test = self.dataclass(
                self.data_dir,
                **self.test_args,  # type: ignore
                **self.setup_args,
                transform=self.transform,
            )
            self.test = self.ds_test


def subset(
    targets: Union[torch.Tensor, np.ndarray],
    ratio: float,
    balanced: bool = True,
    seed: int = 42,
) -> torch.Tensor:
    assert ratio > 0 and ratio <= 1, "Ratio must be between 0 and 1."
    if ratio == 1.0:
        return torch.arange(len(targets))
    numpy_mode = isinstance(targets, np.ndarray)
    if numpy_mode:
        targets = torch.tensor(targets)
    if balanced:
        unique_targets = targets.unique(sorted=True)
        indices = []
        for target in unique_targets:
            class_indices = torch.where(targets == target)[0]
            if len(class_indices) == 0:
                continue
            if len(class_indices) == 1:
                raise ValueError(
                    "Cannot balance dataset with only one sample per class."
                )
            if len(class_indices) == 2:
                size = 1
                warnings.warn(
                    "Balancing dataset with only two samples per class leads to one sample in each split regardless of ratio. (Target: {})".format(
                        target
                    )
                )
            else:
                size = int(ratio * len(class_indices))
            shuffled_class_idx = torch.randperm(
                len(class_indices), generator=torch.Generator().manual_seed(seed)
            )
            indices.append(class_indices[shuffled_class_idx[:size]])
        indices = torch.concatenate(indices)
    else:
        all_indices = torch.arange(len(targets))
        shuffled_idx = torch.randperm(
            len(all_indices), generator=torch.Generator().manual_seed(seed)
        )
        indices = all_indices[shuffled_idx[: int(ratio * len(all_indices))]]
    if numpy_mode:
        indices = indices.numpy()
    return indices


class HuggingFaceViTModel(ViTForImageClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).logits  # type: ignore

    def give_embedding_dimension(self) -> int:
        return self.vit.encoder.layer[-1].output.dense.out_features


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
        }

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50"
            )
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class VIT(LightningModule):
    def __init__(
        self,
        variant: Literal[
            "vit_b_16",
            "vit_l_16",
            "vit_h_14",
            "vit_huge_patch14_224",
            "google/vit-base-patch32-224-in21k",
            "google/vit-huge-patch14-224-in21k",
            "simclr_resnet18",
        ] = "vit_b_16",
        image_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if variant in ["vit_b_16", "vit_l_16", "vit_h_14"]:
            encoder_class, weights_class = {
                "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
                "vit_h_14": (vit_h_14, ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1),
                "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1),
            }[variant]  # default weights as of 09/23
            kwargs = dict(weights=weights_class)
            if image_size is not None:
                kwargs["image_size"] = image_size
            self.encoder = encoder_class(**kwargs)
            emb_dim = self.encoder.heads[0].in_features  # type: int # type: ignore
            self.encoder.heads[0] = torch.nn.Identity()
        elif variant in ["simclr_resnet18_stl10"]:
            model = ResNetSimCLR(base_model="resnet18", out_dim=128)
            ckpt = torch.load(
                "/raid/vit_proto/simclr_resnet18_100_stl10.pth", map_location="cpu"
            )
            model.load_state_dict(ckpt["state_dict"])
            self.encoder = model
            emb_dim = 128
        elif variant in ["simclr_resnet18_cifar10"]:
            model = ResNetSimCLR(base_model="resnet18", out_dim=128)
            ckpt = torch.load(
                "/raid/vit_proto/simclr_resnet18_100_cifar10.pth", map_location="cpu"
            )
            model.load_state_dict(ckpt["state_dict"])
            self.encoder = model
            emb_dim = 128
        elif "/" in variant:
            if image_size is not None:
                raise NotImplementedError(
                    "Image size is not supported for HuggingFace models."
                )
            self.encoder = HuggingFaceViTModel.from_pretrained(variant, num_labels=0)
            emb_dim = self.encoder.give_embedding_dimension()  # type: ignore
        else:
            if image_size is not None:
                raise NotImplementedError(
                    "Image size is not supported for timm models."
                )
            self.encoder = timm.create_model(variant, pretrained=True, num_classes=0)
            emb_dim = None  # type: ignore
            try:
                emb_dim = self.encoder.head.in_features  # type: int # type: ignore
            except AttributeError:
                print(f"Cannot find head for {variant}.")
                try:
                    emb_dim = self.encoder.blocks[-1].mlp.fc2.out_features
                except AttributeError:
                    print(f"Cannot find mlp for {variant}.")
            if emb_dim is None:
                raise ValueError(
                    f"Unsupported model variant {variant}. Can't determine emb_dim."
                )
        self.encoder.eval()  # type: ignore
        for param in self.encoder.parameters():  # type: ignore
            param.requires_grad = False

        self.projection = torch.nn.Identity()  # Optional projection
        self.pool_before_proj = (
            torch.nn.Identity()
        )  # Optional pooling before projection
        self.pool_before_proto = (
            torch.nn.Identity()
        )  # Optional pooling after projection

        self.forward_module = torch.nn.Sequential(
            self.pool_before_proj, self.projection, self.pool_before_proto
        )
        # Print all parameters
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Only project the embeddings. Use for calculating prototypes.

        Args:
            batch (Any): Batch containing the images at index 0.
            batch_idx (int): Not used.
            dataloader_idx (int, optional): Not used.. Defaults to 0.

        Returns:
            torch.Tensor: Projected embeddings.
        """
        if hasattr(self, "encoder"):
            x = self.encoder(batch[0])
        else:
            x = batch[0]
        projected = self.forward_module(x)
        return projected


@dataclass
class DatasetSettings:
    name: str
    ds: str
    data_dir: str = "/raid/datasets"
    num_workers: int = 8
    size_target: int = 224
    center_trick: Optional[int] = None
    min_scale: float = 0.08
    max_scale: float = 1.0
    flip_prob: float = 0.5
    rand_aug_n: int = 0
    rand_aug_m: int = 9
    erase_prob: float = 0.0
    use_trivial_aug: bool = False
    n_way: int = 5
    k_shot: int = 5
    eval_batch_size: int = 512


def give_params(variant):
    eval_batch_size, size_target = {
        "vit_h_14": (256, 518),
        "vit_l_16": (512, 224),
        "vit_b_16": (512, 224),
        "vit_huge_patch14_224.orig_in21k": (256, 224),
        "tiny_vit_21m_224.dist_in22k_ft_in1k": (512, 224),
        "vit_giant_patch14_dinov2.lvd142m": (192, 518),
        "vit_large_patch14_dinov2.lvd142m": (256, 518),
        "google_vit-base-patch32-224-in21k": (1024, 224),
        "google/vit-huge-patch14-224-in21k": (1024, 224),
        "google_vit-huge-patch14-224-in21k": (1024, 224),
        "vit_base_patch16_224.augreg_in21k": (512, 224),
        "vit_large_patch16_224.augreg_in21k": (512, 224),
        "vit_large_patch32_224.orig_in21k": (512, 224),
        "simclr_resnet18": (512, 224),
    }[variant]
    return eval_batch_size, size_target


def sanitize_name(name):
    return name.replace("/", "_")


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices,
            os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"),
        )


def predict_train_dataset(
    dataset_settings: dict, variant_unsafe, out_dir, devices: Iterable[int]
):
    print(
        f"Predicting train ({dataset_settings['name']}, {variant_unsafe}) to {out_dir}"
    )
    try:
        out_dir.mkdir(exist_ok=True, parents=True)
        variant = sanitize_name(variant_unsafe)
        pred_writer = CustomWriter(output_dir=out_dir, write_interval="epoch")
        trainer = Trainer(
            accelerator="gpu",
            devices=devices,
            callbacks=[pred_writer],
        )
        eval_batch_size, size_target = give_params(variant)
        dataset_settings["eval_batch_size"] = eval_batch_size
        center_trick = size_target * 1.5
        dataset_settings["center_trick"] = int(center_trick)
        assert dataset_settings["center_trick"] == int(
            center_trick
        ), "center trick must be integer after scaling"
        dataset_settings["size_target"] = size_target
        model_kwargs = dict(
            variant=variant_unsafe, image_size=dataset_settings["size_target"]
        )
        kwargs = dataset_settings.copy()
        kwargs.pop("name", None)
        if variant_unsafe == "google/vit-huge-patch14-224-in21k":
            image_size = kwargs.pop("size_target", 224)
            if image_size != 224:
                warnings.warn(
                    "google/vit-huge-patch14-224-in21k is used with image size 224"
                )
            kwargs["size_target"] = 224
            model_kwargs.pop("image_size", None)
        model = VIT(**model_kwargs)
        datamodule = ProtoImageData(**kwargs)
        datamodule.setup("fit")
        trainer.predict(
            model,
            datamodule.predict_dataloader(dataset=datamodule.ds_pred),
            return_predictions=False,
        )
        return 0
    except Exception as e:
        print(
            f"Exception during pred_train {dataset_settings['name']} {variant_unsafe}"
        )
        print(e)
        print(traceback.print_exc())


@hydra.main(config_path=".", config_name="config_predict")
def main(cfg: DictConfig):
    OUTDIR = Path(cfg.out_dir)
    datasets = [
        DatasetSettings(
            cfg.dataset_name,
            ds=cfg.dataset_name,
            size_target=cfg.size_target,
            data_dir=cfg.dataset_dir,
        ),
    ]
    variants = cfg.variants

    for ds in datasets:
        (OUTDIR / ds.name).mkdir(exist_ok=True, parents=True)
    for variant, ds in itertools.product(variants, datasets):
        predict_train_dataset(
            dataset_settings=asdict(ds),
            variant_unsafe=variant,
            out_dir=OUTDIR / ds.name / sanitize_name(variant),
            devices=cfg.devices,
        )


if __name__ == "__main__":
    main()
