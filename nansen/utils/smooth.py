import numpy as np


def smooth(x, n=5, window="hanning"):
    """Smooth 1-d data with a window of requested size and type.

    Based on Scipy cookbook:
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    INPUT
    -----
    - `x`: the input signal
    - `n`: the dimension of the smoothing window; should be an odd integer
    - `window`: the type of window ('flat', 'hanning', 'hamming', 'bartlett',
      'blackman'); flat window will produce a moving average smoothing.

    OUTPUT
    ------
    - Smoothed signal of same length as input signal

    EXAMPLE
    -------
    x = np.linspace(0, 7 * np.pi, 100)
    y = np.sin(x - 1) + 0.1 * np.random.randn(n)
    y_smooth = smooth(y, n=11, window='hamming')

    NOTES
    -----
    See also numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve, scipy.signal.lfilter
    """
    smooth_windows = ["flat", "hanning", "hamming", "bartlett", "blackman"]
    if x.size < n:
        raise ValueError("Input vector needs to be larger than window size.")

    if n == 1:  # no need to apply filter
        return x

    if window not in smooth_windows:
        msg = f"Only possible windows: {','.join(smooth_windows)}"
        raise ValueError(msg)

    xadd_left = 2 * x[0] - x[n:0:-1]
    xadd_right = 2 * x[-1] - x[-2 : -n - 2 : -1]
    x_expanded = np.concatenate((xadd_left, x, xadd_right))

    if window == "flat":
        w = np.ones(n, "d")
    else:
        func = getattr(np, window)
        w = func(n)

    y = np.convolve(w / w.sum(), x_expanded, mode="valid")

    i_start = n // 2 + 1
    i_end = i_start + len(x)

    return y[i_start:i_end]
