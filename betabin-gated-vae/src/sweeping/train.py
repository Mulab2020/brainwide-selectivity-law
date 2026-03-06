__all__ = ['train_session', 'train_node']

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.utils import logger
from src.mixing_levels import generate_class_profile
from src.trainer import train_model
from src.dataloader import get_sample_batch
from src.sweeping.session import SweepSession
from src.sweeping.node import NodeInfo
from src.configs import TrainConfig
from src.sweeping.record import TrainRecord
from src.sweeping.summarizer import Summarizer
from src.sweeping.metrics import TrainMetrics
from src.models.utils import get_gated_vae_from_config, get_vae_helpers
from src.visualization.image import show_imgs
from src.visualization.mixing_level_distribution import plot_mld

# =============
# Train Session
# =============

def train_session(
        session: SweepSession,
        train_loader: DataLoader,
        val_loader: DataLoader,
        summary_filename: str = 'train_summary.csv'
) -> None:
    logger.info('Sweeping session launched.')
    
    # sample data
    sample_imgs, sample_lbls = get_sample_batch(val_loader.dataset)
    class_names = val_loader.dataset.classes
    show_imgs(
        sample_imgs, sample_lbls, class_names,
        dst_path=(session.dir / 'sample_ref.png')
    )
    sample_inputs = (
        sample_imgs.to(session.cfg.device),
        sample_lbls.to(session.cfg.device) 
    )
    
    # sweep training
    with Summarizer(session=session, summary_type=TrainMetrics).new(summary_filename) as train_summarizer:
        for node in session.iter_nodes():
            logger.info(
                f'Training {node.id + 1}/{session.n_nodes}: alpha={node.alpha:.3f}, beta={node.beta:.3f}:'
            )
            
            node_train_metrics = train_node(
                node, session.cfg.train,
                train_loader, val_loader, sample_inputs, device=session.cfg.device#, force_replot=True
            )
            train_summarizer.append(node_train_metrics)
    
    logger.info(
        f'Sweeping session completed. Summary saved to {session.dir / summary_filename}.'
    )


# ==========
# Train node
# ==========

def train_node(
        node: NodeInfo,
        train_cfg: TrainConfig,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        sample_inputs: torch.Tensor | None = None,
        force_replot: bool = False,
        device: str = 'cuda'
) -> TrainMetrics:
    
    # control flags
    latest_ckpt_epoch = node.find_latest_ckpt()
    trained_epochs = latest_ckpt_epoch or 0
    n_epochs_to_train = train_cfg.n_epochs - trained_epochs
    
    train_flag = True
    if n_epochs_to_train <= 0:
        train_flag = False
        if n_epochs_to_train == 0:
            logger.info(f'Training for node {node.id} is already completed.')
        else:
            logger.warning(f'Node {node.id} has already been trained for {trained_epochs} epochs.')
    
    _need_model = train_flag or force_replot
    
    # load existing model ckpt if available & necessary
    if _need_model:
        # get model
        if latest_ckpt_epoch is not None:
            # load existing model
            model = get_gated_vae_from_config(node.model_cfg)
            model_path = node.full_ckpt_path(trained_epochs)
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            logger.info(f'Model loaded from {model_path}.')
        else:
            # create new model
            latent_dim = node.model_cfg.vae.latent_dim
            class_profile = generate_class_profile(
                node.n_classes, node.alpha, node.beta, latent_dim, random_control=node.is_control
            )
            model = get_gated_vae_from_config(node.model_cfg, class_profile)
        
        model = model.to(device)
        # adapters
        collate_fn, loss_fn, eval_fn = get_vae_helpers(
            vae_model=model, reconstruction_loss='mse', kld_weight=1.0
        )
    
    # load existing record if available
    if latest_ckpt_epoch is not None:
        # Load existing records
        node_record = TrainRecord.from_npz(node)
    else:
        # create new record
        node_record = TrainRecord(
            mean_activation=model.gate.class_profile.mean().item() # cvae mean activation
        )
    
    if train_flag:
        train_record, val_record = train_model(
            model,
            train_loader,
            loss_fn,
            n_epochs_to_train,
            eval_every_n_epochs = train_cfg.eval_every_n_epochs,
            save_every_n_epochs = train_cfg.save_every_n_epochs,
            val_loader = val_loader,
            eval_fn = eval_fn,
            ckpt_dir = node.ckpt_dir,        
            ckpt_name = 'EP{epoch_id}.pt',
            learning_rate = train_cfg.lr,
            device = device,
            verbose = False,
            collate_fn = collate_fn,
            start_epoch_id = trained_epochs + 1
        )
        
        # Append new records to existing ones
        node_record.extend(train_record, val_record)

        # Save updated training records & probe profiles
        node_record.to_npz(node)
    
    # figs
    if _need_model: # force_replot or train_flag
        model.eval()
        with torch.no_grad():
            # show model sample outputs
            if sample_inputs is not None:
                class_names = train_loader.dataset.classes
                # reconstruction samples
                sample_recs, _, _ = model(sample_inputs)
                sample_labels = sample_inputs[1]
                show_imgs(
                    sample_recs, sample_labels, class_names,
                    dst_path=(node.dir / 'sample_rec.png')
                )
                # generation samples
                gen_imgs, gen_labels = model.generate_images(sample_labels)
                show_imgs(
                    gen_imgs, gen_labels, class_names,
                    dst_path=(node.dir / 'sample_gen.png')
                )
        
        plot_mld(
            class_profile, node.alpha, node.beta,
            dst_path=(node.dir / 'mld.svg'),
        )
        plot_train_losses(node_record.train_loss, (node.dir / 'train_loss.svg'))
        plot_val_record(
            node_record.val_recon, train_cfg.eval_every_n_epochs,
            name = 'reconstruction loss',
            title = 'Reconstruction loss',
            dst_path = (node.dir / 'val_recons.svg'),
            # ylim = (30, 70)
        )
        plot_val_record(
            node_record.val_kld, train_cfg.eval_every_n_epochs,
            name = 'kld loss',
            title = 'KL-divergence loss',
            dst_path = (node.dir / 'val_klds.svg'),
            # ylim = (10, 50)
        )
        plt.close()
    
    return TrainMetrics(
        node_id = node.id,
        alpha = node.alpha,
        beta = node.beta,
        mean_activation = node_record.mean_activation,
        train_loss = node_record.train_loss[-1],
        val_recon = node_record.val_recon[-1],
        val_kld = node_record.val_kld[-1]
    )

# plot utils

def plot_train_losses(train_losses: np.ndarray, dst_path, start_idx: int=10):
    fig = plt.figure()
    ax = fig.gca()
    x = np.arange(len(train_losses))
    ax.plot(x[start_idx:], train_losses[start_idx:], 'k-')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Training loss')
    fig.savefig(dst_path)
    plt.close()

def plot_val_record(record, eval_every_n_epochs, name, title, dst_path, ylim=None, start_idx: int=1):
    fig = plt.figure()
    ax = fig.gca()
    x = np.arange(len(record)) * eval_every_n_epochs
    ax.plot(x[start_idx:], record[start_idx:], 'k-')
    ax.set_xlabel('epoch')
    ax.set_ylabel(name)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.savefig(dst_path)
    plt.close()

