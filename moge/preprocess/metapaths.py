import copy
from typing import Union, Tuple, List, Dict, Any

import numpy as np


def match_groupby_metapaths(groupby_metapaths: Dict[Union[Tuple, str], List[Union[Tuple, str]]],
                            metapaths: List[Tuple[str, str, str]]) \
        -> Dict[Tuple[str, str, str], List[Tuple[str, str, str]]]:
    """
    A dict of target metapath (or etype) and list of source metapaths (or etypes). If an etype string
    is given for target metapath, then expand to all metapaths with matching etype, but if no matches,
    then source and target type will be copied from its source metapath. If an etype string is
    given for source metapath, assume source metapaths have the same source and target node types as target
    metapaths.
    Args:
        groupby_metapaths (): A dict of target metapath (or etype) and list of source metapaths (or etypes).
        metapaths (): A list of canonical metapaths in the hetero network.

    Returns:
        dst_src_expanded_groupby (): A dict of target metapath and list of source metapaths.
    """
    dst_expanded_groupby = {}

    # Expand each target etype to metapaths
    for dst_etype, src_etypes in groupby_metapaths.items():
        if isinstance(dst_etype, str):
            matching_dst_metapaths = [metapath for metapath in metapaths if metapath[1] == dst_etype]
            if not matching_dst_metapaths:
                first_src_etype = next((etype for etype in src_etypes if isinstance(etype, str)), None)
                matching_dst_metapaths = [(head, dst_etype, tail) for head, etype, tail in metapaths \
                                          if etype == first_src_etype]

            for metapath in matching_dst_metapaths:
                dst_expanded_groupby[metapath] = src_etypes

        elif isinstance(dst_etype, tuple):
            dst_expanded_groupby[dst_etype] = src_etypes

        else:
            raise ValueError(f"Invalid metapath {dst_etype}")

    dst_src_expanded_groupby = {}
    # Expand each source etype to metapaths
    for dst_metapath, src_etypes in dst_expanded_groupby.items():
        # replace etype str to matching metapath
        new_src_metapaths = []
        for src_metapath in src_etypes:
            head_tail = {dst_metapath[0], dst_metapath[-1]}
            if isinstance(src_metapath, str):
                src_metapath = next(((head, etype, tail) for head, etype, tail in metapaths \
                                     if etype == src_metapath and {head, tail} == head_tail),
                                    None)

            if isinstance(src_metapath, tuple):
                new_src_metapaths.append(src_metapath)

        if new_src_metapaths:
            dst_src_expanded_groupby[dst_metapath] = new_src_metapaths

    return dst_src_expanded_groupby


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
