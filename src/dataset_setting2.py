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
from dataset_setting import *


def working_in_dist(my_fun, num_total, num_proc, para_ls):
    start = time.time()
    num_split = num_total // num_proc
    output = Parallel(n_jobs=num_proc)(
        delayed(working_in_batch)(my_fun, i, num_total, num_split, num_proc, para_ls) for i in range(num_proc)
    )
    end = time.time()

    total_time = round(end - start, 6)
    print("Learning finished in {} seconds.".format(total_time))

    res = {}
    for i in range(num_proc):
        res.update(output[i])
    return res

def working_in_batch(my_fun, i, num_total, num_split, num_proc, para_ls):
    num_rest = num_total - (i + 1) * num_split
    if (num_rest >= num_split) and (i + 1 < num_proc):
        idx_ls = range(i * num_split, (i+1) * num_split)
    else:
        idx_ls = range(i * num_split, num_total)

    res = {}
    for idx in idx_ls:
        res[idx] = my_fun(para_ls + [idx])
    return res

def str_to_list(str1):
    return [int(num) for num in str1[1:-1].split(',')]


def my_explore_queries_single(para_ls):
    num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode, train_edges, valid_data, valid_data_inv, rel_idx, j = para_ls
    my_model = Walker(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)

    if rel_idx < num_rel//2:
        valid_data_using = valid_data
    else:
        valid_data_using = valid_data_inv

    res = {}
    if valid_data_using[j][1] == rel_idx:
        res = my_model.explore_queries_v2(j, train_edges, valid_data, valid_data_using)
    return res



def my_explore_queries(rel_ls, f_save=True, mode='general'):
    my_model = Walker(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)
    res_dict_total = {}
    for rel_idx in rel_ls:
        if rel_idx < num_rel//2:
            valid_data_using = valid_data
        else:
            valid_data_using = valid_data_inv

        res_dict = {}
        res_dict1 = {}
        for j in range(len(valid_data_using)):
            if valid_data_using[j][1] == rel_idx:
                res = my_model.explore_queries_v3(j, valid_data_using)
                for num in res.keys():
                    for r in res[num]:
                        r = str(r)
                        if r not in res_dict.keys():
                            res_dict[r] = 0
                            res_dict1[r] = []
                        res_dict[r] += 1
                        res_dict1[r].append(j)

        res_dict_total[rel_idx] = res_dict
        if f_save:
            if mode == 'time_shifting':
                with open('output/explore_res/'+dataset_using+'_explore_queries_'+str(rel_idx)+'_t_shift.json', 'w') as f:
                    json.dump(res_dict1, f)
            elif mode in ['general', 'biased']:
                with open('output/explore_res/'+dataset_using+'_explore_queries_'+str(rel_idx)+'.json', 'w') as f:
                    json.dump(res_dict1, f)


    return res_dict_total


def write_query_explore_res(rel_ls, para_ls1, train_edges, valid_data, valid_data_inv, path_name):
    para_ls = para_ls1 + [train_edges, valid_data, valid_data_inv]
    def fun1(x):
        res_dict = {}
        for j in x:
            for num in x[j]:
                for r in x[j][num]:
                    r = str(r)
                    if r not in res_dict.keys():
                        res_dict[r] = []
                    res_dict[r].append(j)
        return res_dict

    test_query_without_sup = []
    test_query_with_sup = []
    sup_train_query = []
    valid_train_query = []

    useful_rules ={}
    for rel_idx in rel_ls:
        res = working_in_dist(my_explore_queries_single, len(valid_data), 24, para_ls + [rel_idx])
        res_dict = fun1(res)

        rule_dict_from_train = {}
        for j in range(len(train_edges)):
            if train_edges[j,1] == rel_idx:
                with open('../output/found_paths/'+dataset_using+'_train_query_'+ str(j) + path_name +'.json') as f:
                    x = json.load(f)
                if len(x) >0:
                    valid_train_query.append(j)
                for num in x:
                    for r in x[num]:
                        if str(r["rule"]) not in rule_dict_from_train.keys():
                            rule_dict_from_train[str(r["rule"])] = []
                        rule_dict_from_train[str(r["rule"])].append(j)

        res_dict_final = []
        useful_rules[rel_idx] = []
        for r in res_dict:
            if r in rule_dict_from_train:
                res_dict_final.append((str_to_list(r), res_dict[r], 1))
                test_query_with_sup += res_dict[r]
                sup_train_query += rule_dict_from_train[r]
                useful_rules[rel_idx].append(str_to_list(r))
            else:
                res_dict_final.append((str_to_list(r), res_dict[r], 0))
                test_query_without_sup += res_dict[r]

    return list(set(sup_train_query)), list(set(valid_train_query)), useful_rules


def add_noise_to_subset(dataset, subset, extra_subset, num_rel, rate):
    a = dataset.tolist()
    b = subset.tolist()
    if len(extra_subset)>0:
        c = extra_subset.tolist()
    else:
        c = []
    noise = [e for e in a if e not in b+c]
    np.random.shuffle(noise)
    noise1 = noise[:int(len(noise)*rate)]
    noise1 = obtain_inv_edges(noise1, num_rel)
    noise1 = np.unique(noise1, axis=0)
    return np.vstack((subset, noise1))


def create_reverse_rules(rule, num_rel):
    r_len = len(rule)//2
    part1 = rule[:r_len][::-1]
    part1 = [num + num_rel//2 if num < num_rel//2 else num - num_rel//2 for num in part1]
    part2 = rule[r_len:][::-1]
    return part1 + part2



def analyze_json_files(path):
    with open(path) as f:
        res_dict = json.load(f)

    res_dict = sorted(res_dict.items(), key = lambda x: len(x[1]), reverse=True)
    for k in res_dict:
        print(k[0], len(k[1]), k[1])


def find_composite_rules():
    def fun0(rel_idx, r_to_check):
        for j in range(len(train_edges)):
            if train_edges[j][1] == rel_idx:
                with open('output/found_paths/'+dataset_using+'_train_query_'+str(j)+'.json') as f:
                    rule_dict = json.load(f)
                if '3' in rule_dict.keys():
                    for r in rule_dict['3']:
                        if r['rule'] == r_to_check:
                            return True
        return False

    def fun1(rel_idx, r_part1, r_part2_f, r_part2_b, t_part2_f, t_part2_b):
        cnt = 0
        res = []
        with open('output/learned_rules/'+dataset_using+'_all_rules_'+str(rel_idx)+'.json') as f:
            check_rules_train_0 = json.load(f)

        y0 = [r['rule'] for r in check_rules_train_0['3']]

        score_dict = {}
        for r in check_rules_train_0['3']:
            score_dict[str(r['rule'])] = r['score']

        for i in range(num_rel):
            res_part2 = []
            for t_p in [-1,0,1]:
                z = r_part2_f + [i] + r_part2_b + t_part2_f +[t_p]+ t_part2_b
                if z in y0:
                    res_part2.append([z, score_dict[str(z)]])

            if len(res_part2)>0:
                res_part2 = sorted(res_part2, key=lambda x: x[1], reverse=True)[0]
                with open('output/learned_rules/'+dataset_using+'_all_rules_'+str(i)+'.json') as f:
                    check_rules_train_x = json.load(f)
                res_part1 = [(r['rule'], r['score']) for r in check_rules_train_x['3'] if r['rule'][:3]==r_part1]
                if len(res_part1)>0:
                    cnt += 1
                    res_part1 = sorted(res_part1, key=lambda x: x[1], reverse=True)[0]
                    res.append([res_part1[0], res_part2[0], res_part1[1]*res_part2[1]])
        if len(res)>0:
            res = sorted(res, key=lambda x: x[2], reverse=True)[0]
        return res


    res_dict = {}
    for rel_idx in range(num_rel//2):
        with open('output/explore_res/'+dataset_using+'_explore_queries_'+str(rel_idx)+'.json') as f:
            check_rules_test = json.load(f)
        check_rules_test = sorted(check_rules_test.items(), key = lambda x1: len(x1[1]), reverse=True)
        for r in check_rules_test:
            r = str_to_list(r[0])
            if len(r)//2 == 5:
                max_score = 0

                x = fun1(rel_idx, r[0:3], r[:0], r[3:5], r[5:5], r[8:10])
                if len(x)>0:
                    if x[-1]>max_score:
                        max_score = x[-1]
                        res = ['type 0'] + x

                x = fun1(rel_idx, r[1:4], r[:1], r[4:5], r[5:6], r[9:10])
                if len(x)>0:
                    if x[-1]>max_score:
                        max_score = x[-1]
                        res = ['type 1'] + x

                x = fun1(rel_idx, r[2:5], r[:2], r[5:5], r[5:7], r[10:10])
                if len(x)>0:
                    if x[-1]>max_score:
                        max_score = x[-1]
                        res = ['type 2'] + x

                if len(res)>0:
                    res[1] = (res[2][int(res[0].split(' ')[-1])], res[1])
                    res[2] = (rel_idx, res[2])
                    res_dict[rel_idx] = {'rule': r, 'results': res}
                    break

    return res_dict


def obtain_divided_dataset(edges, dataset_using):
    if dataset_using == 'YAGO':
        with open('../data/YAGO11k/YAGO_country_ent_dict.json') as f:
            ent_country_dict1 = json.load(f)
    elif dataset_using == 'wiki':
        with open('../data/WIKIDATA12k/wiki_country_ent_dict.json') as f:
            ent_country_dict1 = json.load(f)

    ent_country_dict = {}
    for idx in ent_country_dict1:
        country = ent_country_dict1[idx]
        if country not in ent_country_dict:
            ent_country_dict[country] = []
        ent_country_dict[country].append(int(idx))

    country_list = ent_country_dict.items()
    country_idx_list = [x[1] for x in country_list]
    country_name_list = [x[0] for x in country_list] + ['collaboration']

    def my_divide_dataset(edges, idxset):
        res = []
        residual = np.ones((edges.shape[0], 1))
        for c in idxset:
            y = np.isin(edges[:,2], c)
            x = edges[y]
            residual[y] = 0
            res.append(x)

        residual = residual.reshape(-1)
        return res

    idx_ls = np.array(range(len(edges))).reshape((-1,1))
    edges1 = np.hstack([edges, idx_ls])

    res_train = my_divide_dataset(edges1, country_idx_list)

    return res_train


def pure_random_walk(start_nodes, walk_len, facts1, walk_times):
    idx_ls = np.array(range(len(facts1))).reshape((-1,1))
    facts = np.hstack([facts1, idx_ls])
    res = []
    nodes = copy.copy(start_nodes)
    for i in range(walk_len):
        np.random.shuffle(facts)
        x = facts[np.isin(facts[:,0], nodes)]
        if len(x) == 0:
            break
        res += x[:walk_times].tolist()
        nodes = x[:walk_times][2].tolist()
    return res


def create_query_class():
    if overall_mode == 'time_shifting':
        train_edges, valid_data, valid_data_inv, test_data, test_data_inv = do_temporal_shift_to_dataset(dataset_name1, num_rel)
    elif overall_mode in ['general', 'few', 'biased']:
        train_edges, valid_data, valid_data_inv, test_data, test_data_inv = do_normal_setting_for_dataset(dataset_name1, num_rel)

    with open('../data/YAGO11k/YAGO_country_ent_dict.json') as f:
        ent_country_dict1 = json.load(f)

    ent_country_dict = {}
    for idx in ent_country_dict1:
        country = ent_country_dict1[idx]
        if country not in ent_country_dict:
            ent_country_dict[country] = []
        ent_country_dict[country].append(int(idx))

    country_list = ent_country_dict.items()
    country_idx_list = [x[1] for x in country_list]
    country_name_list = [x[0] for x in country_list]
    print(country_name_list)


    def my_divide_dataset(edges, idxset):
        res = []
        residual = np.ones((edges.shape[0], 1))
        for c in idxset:
            y = np.isin(edges[:,2], c)
            x = edges[y]
            residual[y] = 0
            res.append(x)

        return res

    def obtain_idx_ls(edges, idxset):
        idx_ls = np.array(range(len(edges))).reshape((-1,1))
        x = np.hstack([edges, idx_ls])
        res = my_divide_dataset(x, idxset)
        res_query_class = []
        for x in res:
            res_query_class.append(x[:,-1].tolist())
        return res_query_class

    res_query_class_train = obtain_idx_ls(train_edges[:len(train_edges)//2,:], country_idx_list)
    res_query_class_test = obtain_idx_ls(valid_data, country_idx_list)

    return res_query_class_train, res_query_class_test