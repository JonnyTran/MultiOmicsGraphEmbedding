from argparse import Namespace
from collections.abc import Iterable
from typing import Any, Dict

import numpy as np
import pandas as pd


def get_attrs(obj, exclude: Iterable = None,
              dtypes=(str, dict, int, list, tuple, float, bool,
                      pd.Index, pd.Series, pd.DataFrame, Namespace, np.ndarray)) -> Dict[str, Any]:
    if isinstance(dtypes, Iterable) and not isinstance(dtypes, tuple):
        dtypes = tuple(dtypes)

    attrs = {}
    for name in dir(obj):
        if isinstance(exclude, Iterable) and name in exclude: continue

        val = getattr(obj, name)
        if not callable(val) and (not name.startswith("_") or name == '_name') and isinstance(val, dtypes):
            attrs[name] = val

    return attrs
