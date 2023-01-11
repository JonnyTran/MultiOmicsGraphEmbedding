import copy
from typing import Union, Tuple, List, Dict, Any

import numpy as np


def reverse_metapath(metapath: Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]) \
        -> Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]:
    if isinstance(metapath, list):
        return [reverse_metapath(m) for m in metapath]

    elif isinstance(metapath, dict):
        return {reverse_metapath(m): eid for m, eid in metapath.items()}

    if isinstance(metapath, tuple):
        tokens = []
        for i, token in enumerate(reversed(copy.deepcopy(metapath))):
            if i == 1:
                if len(token) == 2:  # 2 letter string etype
                    rev_etype = token[::-1]
                else:
                    rev_etype = "rev_" + token
                tokens.append(rev_etype)
            else:
                tokens.append(token)

        rev_metapath = tuple(tokens)

        return rev_metapath

    elif isinstance(metapath, str):
        rev_metapath = "".join(reversed(metapath))

    elif isinstance(metapath, (int, np.int)):
        rev_metapath = str(metapath) + "_"
    else:
        raise NotImplementedError(f"{metapath} not supported")


def unreverse_metapath(metapath: Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]) \
        -> Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]:
    if isinstance(metapath, list):
        return [unreverse_metapath(m) for m in metapath]

    elif isinstance(metapath, dict):
        return {unreverse_metapath(m): eid for m, eid in metapath.items()}

    if isinstance(metapath, tuple):
        tokens = []
        for i, token in enumerate(reversed(copy.deepcopy(metapath))):
            if i == 1:
                if len(token) == 2:  # 2 letter string etype
                    rev_etype = token[::-1]
                else:
                    rev_etype = token.removeprefix("rev_")
                tokens.append(rev_etype)
            else:
                tokens.append(token)

        rev_metapath = tuple(tokens)
        return rev_metapath

    else:
        raise NotImplementedError(f"{metapath} not supported")


def is_reversed(metapath):
    if isinstance(metapath, tuple):
        return any("rev_" in token for token in metapath)
    elif isinstance(metapath, str):
        return "rev" in metapath


def tag_negative_metapath(metapath: Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]) \
        -> Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]:
    if isinstance(metapath, list):
        return [tag_negative_metapath(m) for m in metapath]

    elif isinstance(metapath, dict):
        return {tag_negative_metapath(m): eid for m, eid in metapath.items()}

    elif isinstance(metapath, tuple):
        tokens = []
        for i, token in enumerate(copy.deepcopy(metapath)):
            if i == 1:
                if len(token) == 2:  # 2 letter string etype
                    rev_etype = token[::-1]
                else:
                    rev_etype = f"neg_{token}"
                tokens.append(rev_etype)
            else:
                tokens.append(token)

        rev_metapath = tuple(tokens)

        return rev_metapath
    else:
        raise NotImplementedError(f"{metapath} not supported")


def untag_negative_metapath(metapath: Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]) \
        -> Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]:
    if isinstance(metapath, list):
        return [untag_negative_metapath(m) for m in metapath]

    elif isinstance(metapath, dict):
        return {untag_negative_metapath(m): eid for m, eid in metapath.items()}

    elif isinstance(metapath, tuple):
        tokens = []
        for i, token in enumerate(copy.deepcopy(metapath)):
            if i == 1:
                rev_etype = token.removeprefix("neg_")
                tokens.append(rev_etype)
            else:
                tokens.append(token)

        rev_metapath = tuple(tokens)

        return rev_metapath
    else:
        raise NotImplementedError(f"{metapath} not supported")


def is_negative(metapath: Union[Tuple[str, str, str], str]):
    if isinstance(metapath, tuple):
        return any("neg" in token for token in metapath)
    elif isinstance(metapath, str):
        return "neg" in metapath
