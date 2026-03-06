__all__ = ['get_dataloaders', 'get_sample_batch']

from typing import Literal, List, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

def get_dataloaders(
        dataset_name: Literal['EMNIST', 'CIFAR100'],
        batch_size: int,
        pin_memory = True,
        num_workers_train: int = 20,
        num_workers_val: int = 0,
        data_dir: str = 'datasets/',
        mode: Literal['train', 'val', 'both'] = 'both',
        return_n_classes: bool = True
):
    assert mode in ['train', 'val', 'both'], 'mode must be one of ["train", "val", "both"].'
    has_train_data = mode in ['both', 'train']
    has_val_data = mode in ['both', 'val']

    def return_helper(dataloaders: List[DataLoader]):
        if return_n_classes:
            n_classes = len(dataloaders[0].dataset.classes)
            return *dataloaders, n_classes
        else:
            return *dataloaders,

    if dataset_name == 'EMNIST':
        from torchvision.datasets import EMNIST
        from torchvision.transforms import AugMix

        def get_emnist_loader(train: bool):
            _num_workers = num_workers_train if train else num_workers_val
            transform = Compose([AugMix(), ToTensor()]) if train else ToTensor()
            ds = EMNIST(data_dir, split='bymerge', download=True, train=train, transform=transform)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=_num_workers)
            return loader
        
        dataloaders = []
        if has_train_data:
            dataloaders.append(get_emnist_loader(train=True))
        if has_val_data:
            dataloaders.append(get_emnist_loader(train=False))
        return return_helper(dataloaders)
        
    elif dataset_name == 'CIFAR100':
        from torchvision.datasets import CIFAR100
        from torchvision.transforms import RandomHorizontalFlip

        def get_cifar100_loader(train: bool):
            _num_workers = num_workers_train if train else num_workers_val
            transform = Compose([RandomHorizontalFlip(), ToTensor()]) if train else ToTensor()
            ds = CIFAR100(data_dir, train=train, download=True, transform=transform)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=_num_workers)
            return loader
        
        dataloaders = []
        if has_train_data:
            dataloaders.append(get_cifar100_loader(train=True))
        if has_val_data:
            dataloaders.append(get_cifar100_loader(train=False))
        return return_helper(dataloaders)
    
    else:
        raise ValueError(f'Invalid dataset name {dataset_name}.')


# util
def get_sample_batch(dataset: torch.utils.data.Dataset, n_samples: int=None) -> Tuple[torch.Tensor, torch.Tensor]:
    n_classes = len(dataset.classes)
    if n_samples is None:
        n_samples = n_classes
    assert n_samples <= n_classes
    sample_idx = np.full(n_samples, -1, dtype=int)
    _sample_cnt = 0
    for idx, lbl in enumerate(dataset.targets):
        if (lbl < n_samples) and (sample_idx[lbl] == -1):
            sample_idx[lbl] = idx
            _sample_cnt += 1
        if _sample_cnt == n_samples:
            break
    
    imgs = torch.stack([dataset[i][0] for i in sample_idx], dim=0)
    lbls = torch.arange(n_samples, dtype=torch.long)
    return imgs, lbls

