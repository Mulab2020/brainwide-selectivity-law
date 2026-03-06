"""
options:
    -p, --path (str)
        - Path to the probing config file or the sweeping directory.
"""
import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"

from argparse import ArgumentParser

from src.utils import set_global_seed
from src.sweeping import SweepSession, eval_session


if __name__ == '__main__':
    
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="Path to the probing config file or the sweeping directory.")
    args = parser.parse_args()
    
    # prepare sweeping session
    swp = SweepSession(args.path)
    swp.attach()
    
    # for reproducibility
    set_global_seed(swp.cfg.random_seed)
    
    # sweep training
    eval_session(
        session = swp,
        overwrite = False,
        train_summary_filename = 'train_summary.csv',
        eval_summary_filename = 'eval_summary.csv'
    )

