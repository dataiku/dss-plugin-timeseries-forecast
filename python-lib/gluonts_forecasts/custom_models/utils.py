def cast_kwargs(kwargs):
    for arg in kwargs:
        kwargs[arg] = cast_string(kwargs[arg])
    return kwargs


def cast_string(s):
    if s == "True":
        return True
    elif s == "False":
        return False
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
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

DEFAULT_SEASONALITIES = {
    "H": 24,
    "D": 7,
    "W-SUN": 52,
    "W-MON": 52,
    "W-TUE": 52,
    "W-WED": 52,
    "W-THU": 52,
    "W-FRI": 52,
    "W-SAT": 52,
    "M": 12,
    "B": 5,
    "Q-DEC": 4,
}
