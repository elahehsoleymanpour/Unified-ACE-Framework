import numpy as np

def calc_rmse(pred, real): 
    return np.sqrt(np.mean((np.array(pred) - np.array(real))**2))

def calc_qlike(pred, real): 
    return np.mean(np.log(np.array(pred) + 1e-8) + (np.array(real) / (np.array(pred) + 1e-8)))

def calc_dd_curve(cum): 
    peaks = np.maximum.accumulate(cum)
    return (cum - peaks) / peaks
