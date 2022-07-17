from os import listdir
from pathlib import Path
from typing import List, Callable


def list_dir_recursive(root_dir: Path, filter_cb: Callable[[Path], bool] = None):
    """
    List files recursively, going into every subdir of given directory
    :param root_dir: root directory we perform listing for
    :param filter_cb: if given, then it must be a bool callback which
       returns False to skip file
    :return: list of full paths for all items discovered in root_dir given in format
       root_dir / <full item subpath>
    """
    res = []

    def _recursive_body(_parent: Path):
        nonlocal filter_cb, res

        dir_items: List[str] = listdir(str(_parent))
        for di in dir_items:
            di_full = _parent / di
            if di_full.is_dir():
                _recursive_body(di_full)
            else:
                if not filter_cb or filter_cb(di_full):
                    res.append(di_full)

    _recursive_body(root_dir)
    return res



