__all__ = [
    'COLORS',               # palette
    'exportable_plot'            # decorator for saving mathplotlib figures
]

import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Callable, ParamSpec
from functools import wraps

COLORS = {
    'magenta': '#f56098',
    'red': '#fc6262',
    'orange': '#f66f03',
    'yellow': '#cdb718',
    'grass': '#83af00',
    'green': '#05bc59',
    'light-green': '#60BF93',
    'cyan': '#00b6b6',
    'blue': '#00abf5',
    'purple': '#af7cff',
    'pink': '#d56ddd',
    'black': '#000000',
    'gray': '#737373',
}


IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.pdf', '.svg', '.eps'}

def has_img_extension(path: str) -> bool:
    return osp.splitext(path)[1] in IMAGE_EXTENSIONS

P = ParamSpec("P")

def exportable_plot(plot_func: Callable[P, Figure]) -> Callable[P, Figure]:
    @wraps(plot_func)
    def wrapper(*args, **kwargs) -> Figure:
        dst_path = kwargs.pop('dst_path', None)
        bbox_inches = kwargs.pop('bbox_inches', 'tight')
        
        fig = plot_func(*args, **kwargs)
        with plt.rc_context({
            'svg.fonttype': 'none',
            'pdf.fonttype': 42,
            'ps.fonttype': 42
        }):
            if dst_path is not None:
                dst_path = str(dst_path)
                if has_img_extension(dst_path):
                    fig.savefig(dst_path, bbox_inches=bbox_inches)
                else:
                    fig.savefig(dst_path + '.png', bbox_inches=bbox_inches)
                    fig.savefig(dst_path + '.svg', bbox_inches=bbox_inches)
                plt.close()
            else:
                plt.show()
        return fig
    
    return wrapper

