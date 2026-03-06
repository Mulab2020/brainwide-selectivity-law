__all__ = ['get_sorted_cifar100_dataloade', 'GenerativeDataloader']

from typing import Iterable, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from src.models.cvae import ClassAwareGatedVAE


# ================
# Proxy dataloader
# ================

def get_sorted_cifar100_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    train_set: bool = False
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns
    -------
    dataloader: CIFAR-100 DataLoader with samples sorted by class label.
    n_samples: number of samples in the dataset.
    """
    # mkdir_if_not_exists(data_dir)
    dataset = CIFAR100(data_dir, download=True, train=train_set, transform=ToTensor())
    sort_idx = np.argsort(dataset.targets)
    dataset.data = dataset.data[sort_idx]
    dataset.targets = list(np.array(dataset.targets)[sort_idx])
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=num_workers
    )
    dataloader.n_samples = len(dataset)
    return dataloader

class GenerativeDataloader:
    def __init__(
        self,
        gen_model: ClassAwareGatedVAE,
        batch_size: int,
        samples_per_class: int,
        n_classes: int = 100,
        precision_loss: bool = True,
        device: str = 'cuda'
    ):
        """Surrogate dataloader from a given generative model."""
        self.gen_model = gen_model
        self.batch_size = batch_size
        self.labels = torch.arange(n_classes, dtype=torch.long).repeat_interleave(samples_per_class).to(device)
        self.n_samples = n_classes * samples_per_class
        self.n_batches = (self.n_samples - 1) // self.batch_size + 1
        self.n_classes = n_classes
        self.device = device

        if precision_loss:
            # simulate precision loss after a save-load process
            self.sim_precision_loss = lambda x: x.mul_(255).floor_().div_(255)
        else:
            self.sim_precision_loss = lambda x: x
    
    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        self.gen_model.eval()
        with torch.no_grad():
            _start, _end = 0, self.batch_size
            while _start < self.n_samples:
                gen_batch, batch_labels = self.gen_model.generate_images(self.labels[_start:_end], device=self.device)
                _start += self.batch_size
                _end += self.batch_size
                yield self.sim_precision_loss(gen_batch), batch_labels

    def __len__(self):
        return self.n_batches

