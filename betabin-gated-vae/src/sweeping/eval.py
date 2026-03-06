__all__ = ['eval_session']

import json
import numpy as np
import pandas as pd
from torch.nn import Module
from typing import List, Callable, Tuple

from src.utils import logger
from src.configs import FIDEvalConfig
from src.evaluation.fid import FIDEvaluator
from src.evaluation.data import get_sorted_cifar100_dataloader, GenerativeDataloader
from src.sweeping.session import SweepSession
from src.sweeping.summarizer import Summarizer
from src.sweeping.metrics import TrainMetrics, EvalMetrics
from src.sweeping.node import load_node_model, NodeInfo
from src.configs import ModelConfig, FIDEvalConfig
from src.sweeping.record import FIDRecord

# =======
# Helpers
# for calculating rmse from train summary 
# =======

N_PIXEL_VALUES = 32 * 32 * 3 # cifar: 32x32 RGB
def _recon_loss_to_rmse(reconstruction_loss: float | List[float]):
    return np.sqrt(np.array(reconstruction_loss) / N_PIXEL_VALUES)

def _train_summary_to_act_rmse_df(train_summarizer: Summarizer[TrainMetrics]) -> pd.DataFrame:
    """
    SELECT node_id, mean_activation, _recon_loss_to_rmse(val_recon) AS rmse
    FROM train_summary_df
    """
    train_summary_df = train_summarizer.get_data()
    act_rmse_df = train_summary_df.assign(
        rmse=_recon_loss_to_rmse(train_summary_df['val_recon'])
    )[['node_id', 'mean_activation', 'rmse']]
    act_rmse_df.set_index('node_id', inplace=True)
    return act_rmse_df

def make_node_act_rmse_mapper(
        train_summarizer: Summarizer[TrainMetrics]
) -> Callable[[NodeInfo | int], Tuple[float, float]]:
    """
    Returns
    -------
    get_node_act_and_rmse : Callable[[NodeInfo | int], Tuple[float, float]]
        Map a sweep node to its mean_activation and RMSE.
    """
    rmse_df = _train_summary_to_act_rmse_df(train_summarizer)
    def get_node_act_rmse(node: NodeInfo | int) -> Tuple[float, float]:
        if isinstance(node, NodeInfo):
            node = node.id
        return rmse_df.loc[node, ['mean_activation', 'rmse']].values
    return get_node_act_rmse

# ============
# Eval session
# ============

def eval_session(
        session: SweepSession,
        overwrite: bool = False,
        train_summary_filename: str = 'train_summary.csv',
        eval_summary_filename: str = 'eval_summary.csv'
) -> None:
    logger.info('Evaluation session launched.')
    
    # read RMSE
    train_summarizer = Summarizer(session, TrainMetrics).load(train_summary_filename)
    get_node_act_rmse = make_node_act_rmse_mapper(train_summarizer)
    
    # prepare for FID eval
    eval_cfg = session.cfg.fid_eval
    n_classes = session.model_cfg.vae.n_classes
    
    real_loader = get_sorted_cifar100_dataloader(
        data_dir = session.cfg.dataset_dir,
        batch_size = eval_cfg.batch_size,
        num_workers = eval_cfg.num_workers,
        train_set = False
    )
    
    model = None # placeholder
    with Summarizer(session, EvalMetrics).new(eval_summary_filename, overwrite=True) as eval_summarizer, \
        FIDEvaluator(real_loader, n_classes, eval_cfg.inception_dims, session.cfg.device) as evaluator:
            for node in session.iter_nodes():
                logger.info(
                    f'Evaluating {node.id + 1}/{session.n_nodes}: alpha={node.alpha:.3f}, beta={node.beta:.3f}:'
                )
                node_eval_metrics = eval_node(
                    node = node,
                    eval_cfg = eval_cfg,
                    evaluator = evaluator,
                    node_act_rmse_mapper = get_node_act_rmse,
                    model = model,
                    overwrite = overwrite,
                    device = session.cfg.device
                )
                eval_summarizer.append(node_eval_metrics)

    logger.info(
        f'Evaluation session completed. Summary saved to {session.dir / eval_summary_filename}.'
    )


# =========
# Eval node
# =========


def eval_node(
        node: NodeInfo,
        eval_cfg: FIDEvalConfig,
        evaluator: FIDEvaluator,
        node_act_rmse_mapper: Callable[[NodeInfo], Tuple[float, float]],
        model: Module | None = None,
        overwrite: bool = False,
        device: str = 'cuda'
) -> EvalMetrics:
    
    # evaluate FID
    if (not FIDRecord.get_path(node).exists()) or overwrite:
        # eval
        model = load_node_model(
            node=node, ckpt_epochs=node.n_epochs,
            model=model, eval_mode=True, device=device
        )
        gen_loader = GenerativeDataloader(
            gen_model = model,
            batch_size = eval_cfg.batch_size,
            samples_per_class = eval_cfg.samples_per_class,
            n_classes = node.n_classes,
            device = device
        )
        fid_agg, fid_cls = evaluator.evaluate(gen_loader, pbar=True)
        
        # write record
        node_record = FIDRecord(
            aggregated = fid_agg,
            classwise = fid_cls
        )
        node_record.to_npz(node)
    else:
        # load existing FID record
        node_record = FIDRecord.from_npz(node)
    
    # get eval summary
    mean_activation, rmse = node_act_rmse_mapper(node)
    
    return EvalMetrics(
        node_id = node.id,
        alpha = node.alpha,
        beta = node.beta,
        mean_activation = mean_activation,
        rmse = rmse,
        fid = np.mean(node_record.classwise)
    )

