from dku_constants import TIMESERIES_KEYS


def get_model_key(item):
    """Return a unique hashable key for a model that trains on item based on its timeseries identifiers

    Args:
        item (dict): Univariate timeseries dictionary
    """
    if TIMESERIES_KEYS.IDENTIFIERS in item:
        d = item[TIMESERIES_KEYS.IDENTIFIERS].copy()
    else:
        d = {}
    d[TIMESERIES_KEYS.TARGET_NAME] = item[TIMESERIES_KEYS.TARGET_NAME]

    return frozenset(d.items())


def cast_kwargs(kwargs):
    kwargs_copy = kwargs.copy()
    for arg, value in kwargs_copy.items():
        if isinstance(value, dict):
            kwargs_copy[arg] = cast_kwargs(value)
        else:
            kwargs_copy[arg] = cast_string(kwargs_copy[arg])
    return kwargs_copy


def cast_string(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    elif s == "None":
        return None
    elif is_int(s):
        return int(s)
    elif is_float(s):
        return float(s)
    else:
        return s


def is_int(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
