"""
options:
    -p, --path (str)
        - Path to the probing config file or the sweeping directory.
    -c, --continue_training
        - Continue training for more epochs.
    -e, --expand_plan
        - Expand an existing sweeping plan.
"""
import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"

import re
from argparse import ArgumentParser

from src.utils import set_global_seed
from src.dataloader import get_dataloaders
from src.sweeping import SweepSession, train_session


if __name__ == '__main__':
    
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="Path to the probing config file or the sweeping directory.")
    parser.add_argument('-c', '--continue_training', action='store_true', help="Continue training for more epochs.")
    parser.add_argument('-e', '--expand_plan', action='store_true', help="Expand an existing sweeping plan.")
    args = parser.parse_args()
    
    # prepare sweeping session
    swp = SweepSession(args.path)
    if args.continue_training:
        n_epochs = int(input("[Continue training] Train up to (n_epochs: int):\n"))
        swp.set_epochs(n_epochs)
    if args.expand_plan:
        n_expand_nodes = input("[Expand plan] Expand existing plan by (n_nodes: int | Tuple[int, int]):\n")
        n_expand_nodes = re.split(r'[,\s]+', n_expand_nodes)
        n_expand_nodes = list(map(int, n_expand_nodes))
        assert len(n_expand_nodes) <= 2
        swp.expand(n_expand_nodes)
        
    swp.attach()
    
    # for reproducibility
    set_global_seed(swp.cfg.random_seed)
    
    # prepare dataset
    train_loader, val_loader, n_classes = get_dataloaders(
        dataset_name = 'CIFAR100',
        batch_size = swp.cfg.train.batch_size,
        pin_memory = True,
        num_workers_train = swp.cfg.train.num_workers_train,
        num_workers_val = swp.cfg.train.num_workers_val,
        data_dir = swp.cfg.dataset_dir,
        mode = 'both',
        return_n_classes = True
    )
    
    # sweep training
    train_session(
        session = swp,
        train_loader = train_loader,
        val_loader = val_loader,
        summary_filename = 'train_summary.csv'
    )

