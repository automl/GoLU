from tasks.image_classification.large.tiny_vit.data.augmentation.auto_augment import RandAugment, AutoAugment, rand_augment_ops, auto_augment_policy,\
    rand_augment_transform, auto_augment_transform
from tasks.image_classification.large.tiny_vit.data.augmentation.config import resolve_data_config
from tasks.image_classification.large.tiny_vit.data.augmentation.constants import *
from tasks.image_classification.large.tiny_vit.data.augmentation.dataset import ImageDataset, IterableImageDataset, AugMixDataset
from tasks.image_classification.large.tiny_vit.data.augmentation.dataset_factory import create_dataset
from tasks.image_classification.large.tiny_vit.data.augmentation.loader import create_loader
from tasks.image_classification.large.tiny_vit.data.augmentation.mixup import Mixup, FastCollateMixup
from tasks.image_classification.large.tiny_vit.data.augmentation.parsers import create_parser
from tasks.image_classification.large.tiny_vit.data.augmentation.real_labels import RealLabelsImagenet
from tasks.image_classification.large.tiny_vit.data.augmentation.transforms import *
from tasks.image_classification.large.tiny_vit.data.augmentation.transforms_factory import create_transform