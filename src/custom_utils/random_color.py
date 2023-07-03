import numpy as np

_COLORS = np.array(
    [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
        [0.635, 0.078, 0.184],
        [0.300, 0.300, 0.300],
        [0.600, 0.600, 0.600],
        [1.000, 0.000, 0.000],
        [1.000, 0.500, 0.000],
        [0.749, 0.749, 0.000],
        [0.000, 1.000, 0.000],
        [0.000, 0.000, 1.000],
        [0.667, 0.000, 1.000],
        [0.333, 0.333, 0.000],
        [0.333, 0.667, 0.000],
        [0.333, 1.000, 0.000],
        [0.667, 0.333, 0.000],
        [0.667, 0.667, 0.000],
        [0.667, 1.000, 0.000],
        [1.000, 0.333, 0.000],
        [1.000, 0.667, 0.000],
        [1.000, 1.000, 0.000],
        [0.000, 0.333, 0.500],
        [0.000, 0.667, 0.500],
        [0.000, 1.000, 0.500],
        [0.333, 0.000, 0.500],
        [0.333, 0.333, 0.500],
        [0.333, 0.667, 0.500],
        [0.333, 1.000, 0.500],
        [0.667, 0.000, 0.500],
        [0.667, 0.333, 0.500],
        [0.667, 0.667, 0.500],
        [0.667, 1.000, 0.500],
        [1.000, 0.000, 0.500],
        [1.000, 0.333, 0.500],
        [1.000, 0.667, 0.500],
        [1.000, 1.000, 0.500],
        [0.000, 0.333, 1.000],
        [0.000, 0.667, 1.000],
        [0.000, 1.000, 1.000],
        [0.333, 0.000, 1.000],
        [0.333, 0.333, 1.000],
        [0.333, 0.667, 1.000],
        [0.333, 1.000, 1.000],
        [0.667, 0.000, 1.000],
        [0.667, 0.333, 1.000],
        [0.667, 0.667, 1.000],
        [0.667, 1.000, 1.000],
        [1.000, 0.000, 1.000],
        [1.000, 0.333, 1.000],
        [1.000, 0.667, 1.000],
        [0.333, 0.000, 0.000],
        [0.500, 0.000, 0.000],
        [0.667, 0.000, 0.000],
        [0.833, 0.000, 0.000],
        [1.000, 0.000, 0.000],
        [0.000, 0.167, 0.000],
        [0.000, 0.333, 0.000],
        [0.000, 0.500, 0.000],
        [0.000, 0.667, 0.000],
        [0.000, 0.833, 0.000],
        [0.000, 1.000, 0.000],
        [0.000, 0.000, 0.167],
        [0.000, 0.000, 0.333],
        [0.000, 0.000, 0.500],
        [0.000, 0.000, 0.667],
        [0.000, 0.000, 0.833],
        [0.000, 0.000, 1.000],
        [0.000, 0.000, 0.000],
        [0.143, 0.143, 0.143],
        [0.857, 0.857, 0.857],
        [1.000, 1.000, 1.000],
    ]
).astype(np.float32)


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret
