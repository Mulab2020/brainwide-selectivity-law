__all__ = ['SweepSession']

from typing import List, Tuple, Iterator
from pathlib import Path
import numpy as np
import json

from src.utils import logger
from src.configs import (
    SweepConfig, ModelConfig, compare_sweep_configs, get_sweep_config
)
from src.visualization.sweep.heatmap import plot_sweep_heatmap
from src.sweeping.node import NodeInfo

# =============
# Session class
# =============

class SweepSession:
    def __init__(self, sweep_cfg_path: str | Path):
        swp_cfg_path = Path(sweep_cfg_path)
        if not swp_cfg_path.exists():
            raise FileNotFoundError(f'{swp_cfg_path} does not exist.')
        
        if swp_cfg_path.is_dir():
            swp_cfg_path = swp_cfg_path / 'sweep_config.json'
        
        self.cfg: SweepConfig = get_sweep_config(swp_cfg_path)
        with open(self.cfg.model_cfg_path, 'r') as f:
            self.model_cfg = ModelConfig.from_dict(json.load(f))
        
        self.dir = Path(self.cfg.out_dir)
        self.n_nodes = np.prod(self.cfg.n_nodes)
        
        self._attached = False # whether the object is attached to a directory
        
        # plan
        self.node_centers: List[Tuple[float, float]] = None
        self.node_sizes: List[Tuple[float, float]] = None
    

    # public api
    
    @property
    def is_attached(self):
        return self._attached
        
    def attach(self):
        """Attach to the corresponding directory."""
        if self.dir.exists():
            self._sync_existing_dir()    
        else:
            self._create_new_dir()
        self._attached = True
        logger.info(f'Attached to sweeping directory: {self.dir}.')
    
    
    def set_epochs(self, n_epochs: int):
        """Set training epochs."""
        self.cfg.train.n_epochs = n_epochs
        if self._attached:
            self.cfg.save(self.dir / 'sweep_config.json', overwrite=True)
    
    
    def iter_nodes(self) -> Iterator[NodeInfo]:
        self._check_attached()
        for node_id in range(self.n_nodes):
            yield self._prepare_node(node_id)
    
    
    def expand(self, n_expand_nodes: int | Tuple[int, int]):
        """Expand a sweeping plan that already exists.
        +-----------+
        |   area    |
        |    2      |
        +------+----+
        |      |    |
        | area |area|
        | old  | 1  |
        +------+----+
        """
        self._check_attached()
        if not (len(self.cfg.n_nodes) == 2):
            raise ValueError('Expansion not supported for adaptive sweeping plans.')
        if isinstance(n_expand_nodes, int):
            n_expand_nodes = (n_expand_nodes, n_expand_nodes)
        
        # prepare
        n_a, n_b = self.cfg.n_nodes
        a_range_old = self.cfg.range_alpha
        b_range_old = self.cfg.range_beta
        step_a = (a_range_old[1] - a_range_old[0]) / n_a
        step_b = (b_range_old[1] - b_range_old[0]) / n_b

        # expand area 1
        a1 = a_range_old[1] + step_a * (np.arange(n_expand_nodes[0]) + 0.5)
        b1 = b_range_old[0] + step_b * (np.arange(n_b) + 0.5)
        A1, B1 = np.meshgrid(a1, b1)
        node_centers1 = np.stack([A1.ravel(), B1.ravel()], axis=1)

        # expand area 2
        a2 = a_range_old[0] + step_a * (np.arange(n_a + n_expand_nodes[0]) + 0.5)
        b2 = b_range_old[1] + step_b * (np.arange(n_expand_nodes[1]) + 0.5)
        A2, B2 = np.meshgrid(a2, b2)
        node_centers2 = np.stack([A2.ravel(), B2.ravel()], axis=1)
        
        # update config
        self.cfg.n_nodes = [n_a + n_expand_nodes[0], n_b + n_expand_nodes[1]]
        self.cfg.range_alpha = [
            a_range_old[0],
            a_range_old[1] + step_a * n_expand_nodes[0]
        ]
        self.cfg.range_beta = [
            b_range_old[0],
            b_range_old[1] + step_b * n_expand_nodes[1]
        ]
        # update plan
        self.node_centers = np.concatenate([self.node_centers, node_centers1, node_centers2], axis=0)
        base_size = self.node_sizes[0]
        self.node_sizes = np.tile(base_size, (self.node_centers.shape[0], 1))
        # save update
        self._dump()
    
    
    # internal helpers
    
    def _check_attached(self):
        if not self._attached:
            raise RuntimeError('Sweeping object not attached to a directory.')
    
    def _create_new_dir(self):
        self.dir.mkdir(parents=True)
        logger.info(f'Created new sweeping directory: {self.dir}.')
        self.node_centers, self.node_sizes = make_sweep_plan(
            self.cfg.range_alpha,
            self.cfg.range_beta,
            self.cfg.n_nodes
        )
        logger.info('Created sweeping plan.')
        self._dump()
    
    def _sync_existing_dir(self):
        # check sanity
        required = [
            self.dir / 'sweep_config.json',
            self.dir / 'model_config.json',
            self.dir / 'sweep_plan.npz'
        ]
        if not all(p.exists() for p in required):
            raise RuntimeError("Broken sweeping directory")

        # check cfg-cfg match
        with open(self.dir / 'sweep_config.json', 'r') as f:
            cfg = SweepConfig.from_dict(json.load(f))
        self.cfg.model_cfg_path = self.dir / 'model_config.json'
        if not compare_sweep_configs(cfg, self.cfg):
            raise RuntimeError("Sweeping config mismatch")
        
        # check cfg-plan match
        plan = np.load(self.dir / 'sweep_plan.npz')
        self.node_centers = plan['node_centers']
        self.node_sizes = plan['node_sizes']
        if not check_config_plan_match(self.cfg, self.node_centers, self.node_sizes):
            raise RuntimeError("Sweeping plan mismatch")
        
        # update
        self._dump()
        logger.info(f'Sweeping obj synchronized with sweeping directory: {self.dir}.')
    
    def _dump(self):
        model_cfg_path = self.dir / 'model_config.json'
        # save configs
        self.model_cfg.save(self.dir / 'model_config.json', overwrite=True)
        self.cfg.model_cfg_path = model_cfg_path
        self.cfg.save(self.dir / 'sweep_config.json', overwrite=True)
        # save plan
        np.savez(
            self.dir / 'sweep_plan.npz',
            node_centers = self.node_centers,
            node_sizes = self.node_sizes
        )
        # save plan demonstration
        plot_sweep_heatmap(
            self.node_centers, self.node_sizes, title = 'Sweeping Nodes',
            show_grid=True, dst_path = self.dir / 'sweep_plan.png',
        )
        logger.info(f'Updated sweeping configs and plan under directory: {self.dir}.')

    def _prepare_node(self, node_id: int):
        # node_dir = self.dir / str(node_id)
        node_dir = self.dir / f'{node_id:03d}'
        
        alpha, beta = self.node_centers[node_id]
        return NodeInfo(
            id = node_id,
            dir = node_dir,
            alpha = alpha,
            beta = beta,
            n_classes = self.model_cfg.vae.n_classes,
            is_control = self.cfg.is_control,
            n_epochs = self.cfg.train.n_epochs,
            model_cfg = self.model_cfg
        )

# =======
# Helpers
# =======

def check_config_plan_match(
    cfg: SweepConfig,
    node_centers: np.ndarray,
    node_sizes: np.ndarray
) -> bool:
    if len(node_centers) != len(node_sizes):
        logger.warning('Broken sweeping plan: node center and size mismatch.')
        return False
    if np.prod(cfg.n_nodes) != len(node_centers):
        logger.warning('Sweeping config and plan mismatch in node count.')
        return False
    
    _half_node_sizes = node_sizes / 2
    alpha_max, beta_max = (node_centers + _half_node_sizes).max(axis=0)
    alpha_min, beta_min = (node_centers - _half_node_sizes).min(axis=0)
    
    if not (
        np.allclose(cfg.range_alpha, [alpha_min, alpha_max]) and
        np.allclose(cfg.range_beta, [beta_min, beta_max])
    ):
        logger.warning('Sweeping config and plan mismatch in range.')
        return False
    
    return True

# planner

def make_sweep_plan(
        alpha_range: Tuple[float, float],
        beta_range: Tuple[float, float],
        n_nodes: Tuple[int, int]
):
    """Plan sweeping parameter coordinates.
    """
    n_a, n_b = n_nodes
    a_min, a_max = alpha_range
    b_min, b_max = beta_range

    a = (a_max - a_min) * (np.arange(n_a) + 0.5) / n_a + a_min
    b = (b_max - b_min) * (np.arange(n_b) + 0.5) / n_b + b_min

    A, B = np.meshgrid(a, b)
    node_centers = np.stack([A.ravel(), B.ravel()], axis=1)
    tile_size = (a[1] - a[0], b[1] - b[0])
    node_sizes = np.array([tile_size,] * (n_nodes[0] * n_nodes[1]))

    return node_centers, node_sizes

