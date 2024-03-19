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

from Models import TILP

import tensorflow as tf


def my_train(i, num_relations, num_processes):
    num_rest_relations = num_rel1 - (i + 1) * num_relations
    if (num_rest_relations >= num_relations) and (i + 1 < num_processes):
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, num_rel1)

    loss_dict = {}
    my_trainer = TILP(num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode)

    for rel_idx in relations_idx:
        loss_dict[rel_idx] = {}
        for ruleLen in range(1, self.num_ruleLen+1):
            if ruleLen in num_paths_dict[rel_idx].keys():
                loss_dict[rel_idx][ruleLen] = []
                
                cur_num_train = 1
                if (num_paths_dict[rel_idx][ruleLen] > my_trainer.num_paths_max) or \
                        (num_samples_dict[rel_idx][ruleLen] > my_trainer.num_training_samples):
                    cur_num_train = 3
                
                for train_idx in range(cur_num_train):
                    targ_path = my_trainer.get_weights_savepath(rel_idx, ruleLen, train_idx)
                    if not os.path.exists(targ_path):
                        print('training rel ' + str(rel_idx) + ', len ' + str(ruleLen) + ', idx ' + str(train_idx))
                        loss = my_trainer.train_single_rel_ver2(rel_idx, ruleLen, train_idx)

                        loss_dict[rel_idx][ruleLen].append(loss)
    return loss_dict


def my_train_v2(i, num_relations, num_processes, rel_empty_ls, mode, para_ls_for_trainer,
                     train_edges, const_pattern_ls,
                    train_idx = None, path_name = ''):
    num_rel1 = len(rel_empty_ls)
    num_rest_relations = num_rel1 - (i + 1) * num_relations
    if (num_rest_relations >= num_relations) and (i + 1 < num_processes):
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, num_rel1)

    loss_dict = {}
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    my_trainer = TILP(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)

    num_trainMax_dict = [(0,300,3000),(300,1000,1200),(1000,3000,400),(3000,100000,120)]

    for rel_idx1 in relations_idx:
        rel_idx = rel_empty_ls[rel_idx1]
        if mode == 'general':
            cur_num_samples = len(train_edges[train_edges[:,1]==rel_idx])
        elif 'few' in mode:
            x = train_edges
            cur_num_samples = len(x[x[:,1]==rel_idx])
        num_trainMax = [x[-1] for x in num_trainMax_dict if (cur_num_samples>=x[0]) and (cur_num_samples<x[1])][0]
        loss = my_trainer.train_single_rel_ver4(rel_idx, num_trainMax, train_edges, const_pattern_ls, mode, train_idx, path_name)
        loss_dict[rel_idx] = loss.tolist()

    return loss_dict


def my_train_v3(i, num_relations, num_processes, rel_empty_ls, mode, para_ls_for_trainer,
                     train_edges, const_pattern_ls,
                    train_idx = None, path_name = ''):
    num_rel1 = len(rel_empty_ls)
    num_rest_relations = num_rel1 - (i + 1) * num_relations
    if (num_rest_relations >= num_relations) and (i + 1 < num_processes):
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, num_rel1)

    loss_dict = {}
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    my_trainer = TILP(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)

    num_trainMax_dict = [(0,300,3000),(300,1000,1200),(1000,3000,400),(3000,100000,120)]

    for rel_idx1 in relations_idx:
        rel_idx = rel_empty_ls[rel_idx1]
        if mode == 'general':
            cur_num_samples = len(train_edges[train_edges[:,1]==rel_idx])
        elif 'few' in mode:
            x = train_edges
            cur_num_samples = len(x[x[:,1]==rel_idx])
        num_trainMax = [x[-1] for x in num_trainMax_dict if (cur_num_samples>=x[0]) and (cur_num_samples<x[1])][0]
        loss = my_trainer.train_single_rel_ver5(rel_idx, num_trainMax, train_edges, const_pattern_ls, mode, train_idx, path_name)
        loss_dict[rel_idx] = loss.tolist()

    return loss_dict


def my_train_v4(rel_empty_ls, mode, para_ls_for_trainer,
                     train_edges, const_pattern_ls,
                    train_idx = None, path_name = ''):
    num_epoch = 200
    loss_dict = {}
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    my_trainer = TILP(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)
    loss = my_trainer.TRL_model_training_v2(rel_empty_ls, num_epoch, train_edges, const_pattern_ls, mode, train_idx, path_name)
    loss_dict['loss'] = [l.tolist() for l in loss]

    return loss_dict



def do_my_train_v2(para_ls_for_trainer, train_edges, const_pattern_ls, 
                    mode='general', rel_empty_ls = None, train_idx = None, path_name=''):
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    if rel_empty_ls == None:
        rel_empty_ls = [i for i in range(num_rel)]
    num_rel1 = len(rel_empty_ls)
    num_processes = min(24, len(rel_empty_ls)) # 1 for test


    start = time.time()
    num_relations = num_rel1 // num_processes
    output = Parallel(n_jobs=num_processes)(
            delayed(my_train_v2)(i, num_relations, num_processes, rel_empty_ls, mode, para_ls_for_trainer,
                            train_edges, const_pattern_ls, train_idx, path_name) for i in range(num_processes)
    )
    end = time.time()

    total_time = round(end - start, 6)
    print("Learning finished in {} seconds.".format(total_time))

    loss_dict = {}

    for i in range(num_processes):
        loss_dict.update(output[i])

    with open('../output/' + dataset_using +'_loss_dict.json', 'w') as f:
        json.dump(loss_dict, f)


def do_my_train_v3(para_ls_for_trainer, train_edges, const_pattern_ls, 
                    mode='general', rel_empty_ls = None, train_idx = None, path_name=''):
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    if rel_empty_ls == None:
        rel_empty_ls = [i for i in range(num_rel)]
    num_rel1 = len(rel_empty_ls)
    num_processes = min(24, len(rel_empty_ls)) # 1 for test


    start = time.time()
    num_relations = num_rel1 // num_processes
    output = Parallel(n_jobs=num_processes)(
            delayed(my_train_v3)(i, num_relations, num_processes, rel_empty_ls, mode, para_ls_for_trainer,
                            train_edges, const_pattern_ls, train_idx, path_name) for i in range(num_processes)
    )
    end = time.time()

    total_time = round(end - start, 6)
    print("Learning finished in {} seconds.".format(total_time))

    loss_dict = {}

    for i in range(num_processes):
        loss_dict.update(output[i])

    with open('../output/' + dataset_using +'_loss_dict.json', 'w') as f:
        json.dump(loss_dict, f)



def do_my_train_TRL(para_ls_for_trainer, train_edges, const_pattern_ls, 
                    mode='general', targ_rel_ls = None, pos_examples_idx = None, path_name=''):
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    if targ_rel_ls == None:
        targ_rel_ls = [i for i in range(num_rel)]

    start = time.time()
    loss_dict = my_train_v4(targ_rel_ls, mode, para_ls_for_trainer,
                            train_edges, const_pattern_ls, pos_examples_idx, path_name)
    end = time.time()
    total_time = round(end - start, 6)
    print("Learning finished in {} seconds.".format(total_time))

    with open('../output/' + dataset_using +'_loss_dict.json', 'w') as f:
        json.dump(loss_dict, f)


def do_my_train_tfm(para_ls_for_trainer, rel_ls, train_edges, dist_pars):
    num_epoch = 100
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    my_trainer = TILP(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)
    my_trainer.train_tfm_v2(rel_ls, num_epoch, train_edges, dist_pars)


def do_my_train_tfm_Wc(para_ls_for_trainer, rel_ls, train_edges, dist_pars):
    num_epoch = 100
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode = para_ls_for_trainer
    my_trainer = TILP(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)
    my_trainer.train_tfm_Wc_v2(rel_ls, num_epoch, train_edges, dist_pars)