import numpy as np

# interpolation function
def interpolate(data):
    for col in data.columns:
        data[col].interpolate(inplace=True, limit_direction='both', limit = 10)
    return data

def nandivide(a, b):
    if np.isscalar(b):
        if b==0:
            return np.nan
    elif np.any(b==0):
        out = a/b
        out[np.isinf(out)] = np.nan
        return out
    return a/b

def rle(inarray, dt=None):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.array(inarray, dtype=np.int32)                  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions

        if dt is None:
            return z, p, ia[i] # simply return array runlengths
        else:
            try:
                dt = np.array(dt)   # force numpy
                l = np.zeros(z.shape) ## real time durations
                for j,_ in enumerate(p[:-1]):
                    l[j] = np.sum(dt[p[j]:p[j+1]])
                l[-1] = np.sum(dt[p[-1]:]) ## length of last segment
                return z, p, ia[i], l # return array runlengths & real time durations
            except TypeError:
                print('Your array is invalid')

def savgolfilt(series, order=3, window=5):
    """
    Returns filtered series based on SciPy's SavGol filter
    """
    from scipy.signal import savgol_filter
    filt = series
    if sum(~np.isnan(filt)) > window:
        filt[~np.isnan(filt)] = savgol_filter(filt[~np.isnan(filt)], window, order)
    return filt
