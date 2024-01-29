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

from Models import Trainer


def my_create_all_rule_dicts(i, n_s, n_p, rel_idx, train_edges, para_ls_for_trainer, mode='total', 
                                pos_examples_idx= None):
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer

    n_t = len(train_edges)//2
    n_r = n_t - (i + 1) * n_p
    s = 0
    if rel_idx >= num_rel//2:
        s = len(train_edges)//2

    if (n_r >= n_s) and (i + 1 < n_p):
        idxs = range(s+i * n_s, s+(i + 1) * n_s)
    else:
        idxs = range(s+i * n_s, s+n_t)

    my_trainer = Trainer(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)
    rule_dict, rule_sup_num_dict = my_trainer.create_all_rule_dicts(rel_idx, train_edges, idxs, mode, 
                                                                        pos_examples_idx)
    return rule_dict, rule_sup_num_dict


def do_my_create_all_rule_dicts(this_rel, train_edges, para_ls_for_trainer, mode='total', 
                                pos_examples_idx=None):
    n_p = 24
    start = time.time()
    n_s = (len(train_edges)//2) // n_p
    output = Parallel(n_jobs=n_p)(
        delayed(my_create_all_rule_dicts)(i, n_s, n_p, this_rel, train_edges, 
                                            para_ls_for_trainer, mode, 
                                            pos_examples_idx) for i in range(n_p)
    )
    end = time.time()

    total_time = round(end - start, 6)
    print("Learning finished in {} seconds.".format(total_time))


    rule_sup_num_dict = {}
    rule_dict = {}
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    my_trainer = Trainer(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)
    for i in range(n_p):
        rule_sup_num_dict = my_trainer.my_merge_dict(rule_sup_num_dict, output[i][1])
        for k in output[i][0].keys():
            if k not in rule_dict.keys():
                rule_dict[k] = output[i][0][k].copy()
            else:
                rule_dict[k] = np.vstack((rule_dict[k], output[i][0][k]))
            rule_dict[k] = np.unique(rule_dict[k], axis=0)

    for k in rule_dict.keys():
        print('ruleLen', str(k), 'ruleShape', rule_dict[k].shape)

    return rule_dict, rule_sup_num_dict


def do_calculate_rule_scores(rel_ls, para_ls_for_trainer, 
                                train_edges, const_pattern_ls, 
                                mode='total', pos_examples_idx=None):
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    my_trainer = Trainer(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)
    for rel in rel_ls:
        rule_dict, rule_sup_num_dict = do_my_create_all_rule_dicts(rel, train_edges, para_ls_for_trainer, 
                                                                    mode, pos_examples_idx)
        my_trainer.write_all_rule_dicts(rel, rule_dict, rule_sup_num_dict, train_edges, const_pattern_ls, mode)


def explore_all_rules_dict(rel_idx):
    with open('output/learned_rules/' + dataset_using +'_all_rules_'+str(rel_idx)+'.json','r') as f:
        rule_dict1 = json.load(f)

    for k in rule_dict1.keys():
        print(k, len(rule_dict1[k]))
        # x = [r['score'] for r in rule_dict1[k] if r['score']>1e-30]
        # print(min(x), max(x), len(x))
        x = np.array([r['rule'] for r in rule_dict1[k]])
        print(x.shape)
        y = np.unique(x, axis=0)
        print(y.shape)
        y = np.unique(x[:,:int(k)], axis=0)
        print(y.shape)
        y = np.unique(x[:,:int(k)-1], axis=0)
        print(y.shape)
        print(' ')