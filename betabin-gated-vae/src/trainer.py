__all__ = ['train_model']

import numpy as np
from tqdm import tqdm
import torch
from typing import Optional, Callable, Tuple, Any
# import os.path as osp
from pathlib import Path

from src.utils import logger

def default_collate_fn(batch_input: torch.Tensor, batch_target: torch.Tensor, device):
    return batch_input.to(device), batch_target.to(device)


def train_epoch(
        model,
        train_loader,
        optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], float],
        device: str = 'cuda',
        use_pbar = True,
        pbar_desc: Optional[str] = None,
        collate_fn: Optional[Callable[[torch.Tensor, torch.Tensor, str], Tuple[Any, Any]]] = default_collate_fn
):
    """
    Train the model for one epoch.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Yields (batch_input, batch_target) tuples.
        `batch_target` could have shape (batch_size, target_dim) or (batch_size,).
    loss_fn : Callable
        Feeds (batch_output, batch_target). Returns `loss`.
    device : str, default='cuda'
    use_pbar : bool, optional
        Whether to use a progress bar (default is True).
    collate_fn : Callable
        Feeds (batch_input, batch_target, device). Returns collated `(batch_input, batch_target)`.
    
    Returns
    -------
    avg_loss : float
        Averaged loss over the training epoch.
    """
    model.train()
    loss_sum = 0
    
    if use_pbar:
        train_iterator = tqdm(train_loader, desc=pbar_desc)
    else:
        train_iterator = train_loader
    
    for curr_iter, (batch_input, batch_target) in enumerate(train_iterator, 1):
        optimizer.zero_grad()
        batch_input, batch_target = collate_fn(batch_input, batch_target, device)
        
        output = model(batch_input)
        loss = loss_fn(output, batch_target)
        
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        
        if use_pbar:
            train_iterator.set_postfix(avg_loss = loss_sum / curr_iter)
    
    return loss_sum / curr_iter


def eval_model(
        model,
        val_loader,
        eval_fn: Callable[[torch.Tensor, torch.Tensor], float],
        device: str = 'cuda',
        use_pbar = True,
        collate_fn: Optional[Callable[[torch.Tensor, torch.Tensor, str], Tuple[Any, Any]]] = default_collate_fn
):
    """
    Evaluate the model on a validation set.
    
    Parameters
    ----------
    eval_fn : Callable
        Feeds (batch_output, batch_target). Returns `eval metric(s)` (float or np.ndarray).
    collate_fn : Callable
        Feeds (batch_input, batch_target, device). Returns collated `(batch_input, batch_target)`.
    """
    model.eval()
    with torch.no_grad():
        eval_sum = 0
        val_iterator = tqdm(val_loader) if use_pbar else val_loader
        for curr_iter, (batch_input, batch_target) in enumerate(val_iterator, 1):
            batch_input, batch_target = collate_fn(batch_input, batch_target, device)
            
            output = model(batch_input)
            # support both scalar and vector metrics:
            eval_sum = eval_sum + eval_fn(output, batch_target)
            
            if use_pbar:
                val_iterator.set_postfix({'avg_eval(s)': np.divide(eval_sum, curr_iter)})
    # use `np.divide` instead of `/` to skip inf or nan as they may converge later in training:
    return np.divide(eval_sum, curr_iter)


def train_model(
        model,
        train_loader: torch.utils.data.DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], float],
        n_epochs,
        eval_every_n_epochs,
        save_every_n_epochs,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        eval_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        ckpt_dir = None,
        ckpt_name = 'EP{epoch_id}.pt',
        optimizer = None,
        learning_rate = None,
        verbose = True,
        device: str = 'cuda',
        collate_fn: Optional[Callable[[torch.Tensor, torch.Tensor, str], Tuple[Any, Any]]] = default_collate_fn,
        start_epoch_id: int = 1
):
    """
    Trains a model using the provided data loaders and parameters.

    Parameters
    ----------
    loss_fn : callable
        Feeds (batch_output, batch_target). Returns `loss`.
    eval_fn : callable
        Feeds (batch_output, batch_target). Returns `eval metric(s)` (float or np.ndarray).
    save_every_n_epochs : int, default=0
        Save the model every `save_every_n_epochs` epochs. If 0, no model will be saved.
    verbose : bool, default=True
        Progress bars are set for each epoch if verbose=True.
        Otherwise, only one progress bar will be shown globally.
    collate_fn : Callable
        Feeds (batch_input, batch_target, device). Returns collated `(batch_input, batch_target)`.
    ckpt_dir : str, default=None
        Directory to save checkpoints.
    ckpt_name : str, default='EP{epoch_id}.pt'
        Format string for checkpoint file names.
    Returns
    -------
    tuple[np.ndarray, np.ndarray] :
        Training losses and validation metrics.
    """
    train_loss_record = []
    val_metrics_record = []
    
    if optimizer is None:
        assert learning_rate is not None, 'learning_rate must be provided when optimizer not specified.'
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=learning_rate)
    
    save_flag = (save_every_n_epochs > 0) and ckpt_dir
    if save_flag:
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f'{ckpt_dir} does not exist.')
    else:
        logger.warning('No checkpoints will be saved: `ckpt_dir` not specified or `save_every_n_epochs <= 0`.')
    
    # for progress bar
    if not verbose: # enable global tqdm, but the nested ones would be disabled later:
        epoch_iterator =  tqdm(range(start_epoch_id, n_epochs + start_epoch_id), desc='Training')
        if eval_every_n_epochs > 0:
            val_metrics_record.append(
                eval_model(model, val_loader, eval_fn, device, verbose, collate_fn)
            )
            epoch_iterator.set_postfix(
                {'eval(s)': val_metrics_record[-1]}
            )
        else:
            val_metrics_record.append(-1)
    else: # disable global tqdm, but the nested ones would be enabled later:
        epoch_iterator = range(1, n_epochs + 1)
    
    for epoch_id in epoch_iterator:
        train_loss_record.append(
            train_epoch(
                model, train_loader, optimizer, loss_fn, device, verbose,
                f'Epoch {epoch_id}/{n_epochs}', collate_fn
            )
        )
        
        if (eval_every_n_epochs > 0) and (epoch_id % eval_every_n_epochs == 0):
            val_metrics_record.append(
                eval_model(model, val_loader, eval_fn, device, verbose, collate_fn)
            )
        
        if save_flag and (epoch_id % save_every_n_epochs == 0):
            torch.save(model.state_dict(), ckpt_dir / ckpt_name.format(epoch_id=epoch_id))
        
        if not verbose:
            epoch_iterator.set_postfix({
                'train_loss': train_loss_record[-1],
                'eval(s)': val_metrics_record[-1]
            })
    
    if save_flag:
        torch.save(model.state_dict(), ckpt_dir / ckpt_name.format(epoch_id=n_epochs+start_epoch_id-1) )
    
    if start_epoch_id > 1:
        logger.info(f'`start_epoch_id` > 1, first validation record before training is discarded.')
        return np.array(train_loss_record), np.array(val_metrics_record)[1:]
    return np.array(train_loss_record), np.array(val_metrics_record)

