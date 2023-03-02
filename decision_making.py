import numpy as np
import pandas as pd
from numpy import random
from operator import itemgetter
import copy
import time
import itertools
from multiprocessing import Pool
from functools import partial
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns

import os
import csv
import json

import exp_wrapper 
import experiment_helper
import finite_sim
import finite_sim_wrapper

# Takes in individual samples or dataframe of point_est and se_est
# Outputs (dataframe of) dictionary
def confidence_interval(confidence_level, point_est, se_est):
    alpha = 1-confidence_level
    z = norm.ppf(1-alpha/2)
    lower = point_est - z*se_est
    upper = point_est + z*se_est
    # if isinstance(point_est, pd.DataFrame):
    #     return pd.DataFrame({col: zip(lower[col], upper[col]) for col in lower.columns})
    # else:
    return {'lower':lower, 'upper':upper}

def contained_in_interval(interval_dict,point):
    return (interval_dict['lower'] <= point) & (point <= interval_dict['upper'])
    
def acceptance_outcome_at_threshold(interval_dict, threshold):
    return threshold <= interval_dict['lower']

    
# For a list of intervals, calculates number of acceptances and rejections
def p_accept_at_threshold(intervals, threshold):
    # outcomes = []
    # for interval in intervals_list:
    #     outcomes.append(acceptance_outcome_at_threshold(interval, threshold))
    
    outcomes = (threshold <= intervals['lower'])
    

    #n_reject = len(outcomes) - n_accept
    return outcomes.sum()/len(outcomes)

def freq_contained_in_intervals(intervals, point):
    contained = (intervals['lower'] <= point) & (point <= intervals['upper'])

    return contained.sum() / len(contained)
