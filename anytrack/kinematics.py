import numpy as np
import os
import os.path as op
import pandas as pd

### internals
#from defs import *
from anytrack.signal_proc import interpolate, savgolfilt

def get_angle(_data_origin, _data_tip):
    """
    Returns angular heading for given origin and tip positions in DEGREES.
    """
    base, tip = np.array(_data_origin), np.array(_data_tip)
    dx, dy = tip[:,0]-base[:,0], tip[:,1]-base[:,1]
    return np.arctan2(dy,dx)

# Calculate walking velocities
def get_translational_speed(data, dt=1, scale=1):
    """
    Returns translational speed from x,y dataframe and dt vector
    """
    arr = np.array(data)
    _dt = np.array(dt)
    x,y = arr[:,0],arr[:,1]
    ### linear speed is the squareroot of squared displacements in x and y (Pythagoras' theorem) divided by dt
    dr = np.hypot( np.diff(x), np.diff(y) )
    dr = np.divide(np.append(dr[0], dr), scale)
    speed = np.divide(dr,_dt)
    ### filter speeds once
    speed = savgolfilt(speed)
    return speed

def get_rotational_speed(angle, dt=1):
    """
    Returns translational speed from angle
    """
    da = np.append(0, np.diff(angle))
    da[da>np.pi] -= 2*np.pi  ## correction for circularity
    da[da<-np.pi] += 2*np.pi  ## correction for circularity
    rotspeed = np.divide(da,dt)
    ### filter speeds once
    rotspeed = savgolfilt(rotspeed)
    return rotspeed

def angle2pi(angle):
    """
    Radians defined from 0 to 2*pi TODO: Is this needed?
    """
    angle[angle<0] += 2*np.pi
    return angle

def get_distance_to_center(_data):
    """
    Returns distances to center
    """
    dist_sq = np.square(_data.iloc[:,0]) + np.square(_data.iloc[:,1])
    return np.sqrt(dist_sq)

def get_distance_to_patch(_data, _patch):
    """
    Returns distances to each food spot
    """
    xp, yp = _patch["x"], _patch["y"]
    dist_sq = np.square(_data.iloc[:, 0] - xp) + np.square(_data.iloc[:, 1] - yp)
    return np.sqrt(dist_sq)

def get_distances(headPos, center=None, spots=None, px_per_mm=None):
    """
    Returns dataframe of all distances from head position to center and spots
    """
    if spots is None: spots = []
    distances = pd.DataFrame()
    distances['distance_center'] = get_distance_to_center(headPos)
    for i, spot in enumerate(spots):
        ## center around origin
        spot['x'] -= center[0]
        spot['y'] -= center[1]
        ## scale to mm & flip y
        spot['x'] /= px_per_mm
        spot['y'] /= -px_per_mm
        ## distance to spot i
        distances['distance_patch_{}'.format(i)] = get_distance_to_patch(headPos, spot)
    return distances

def get_consttimepoints(x, y):
    """
    Returns constant timepoints based on path length for curvature
    """
    n = len(x)
    t = np.arange(n)
    p = np.hstack((0,np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    pp = np.linspace(0, p[-1], n,endpoint=True)
    # interpolate time based on path length
    tp = np.interp(pp, p, t)
    return t, tp

def resample_path(x, y, t, tp):
    '''
    Resample 2D path such that points are separated by constant translational distance
    '''
    #interpolate positions at times that correspond to equal path movements
    xp = np.interp(tp, t, x)
    yp = np.interp(tp, t, y)
    return xp, yp

def get_curvature(_data):
    '''
    Returns signed curvature
    '''
    # get x, y positions
    x,y = _data.iloc[:,0].values,_data.iloc[:,1].values
    # compute constant spaced time points based on path length
    t, tp = get_consttimepoints(x, y)
    # compute resampled path based on spaced time points
    xp, yp = resample_path(x, y, t, tp)
    # compute curvature
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)
    d2x = np.gradient(dx) #second derivatives
    d2y = np.gradient(dy)
    curvature = d2y/((1 + dy**2)**1.5)
    curvature = savgolfilt(curvature, window=7) #smooth
    return np.interp(t, tp, curvature)
