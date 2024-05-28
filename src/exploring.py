import random
import numpy as np
import sys
import json
from joblib import Parallel, delayed
import time
import os
import pandas as pd
import copy
from collections import Counter
from scipy.stats import norm

from Models import *


def do_explore_queries():
    my_model = TILP(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)

    res = my_model.explore_queries_v2()

    for k in res.keys():
        print(k, res[k])

def my_explore(i, num_queries, num_processes):
    my_model = TILP(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)
    my_model.explore_in_batch(i, num_queries, num_processes)