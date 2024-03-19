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



def my_create_rule_supplement(i, n_q, n_p, rel_idx):
    my_model = TILP(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)
    q_idx = my_model.create_training_idx_in_batch(i, n_q, n_p, rel_idx)
    rule_sup = my_model.create_rule_supplement(rel_idx, q_idx)
    return rule_sup


def do_my_create_rule_supplement(rel_idx):
    n_t = len(train_edges)//2
    n_p = 24

    start = time.time()
    n_q = n_t // n_p
    output = Parallel(n_jobs=n_p)(
        delayed(my_create_rule_supplement)(i, n_q, n_p, rel_idx) for i in range(n_p)
    )
    end = time.time()

    total_time = round(end - start, 6)
    print("Learning finished in {} seconds.".format(total_time))

    rule_sup = output[0]
    my_model = TILP(num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode)

    for i in range(1, n_p):
        for k in rule_sup.keys():
            rule_sup[k] = my_model.my_merge_dict(rule_sup[k], output[i][k])

    for k in rule_sup.keys():
        s = sum(rule_sup[k].values())
        rule_sup[k] = [{'rule': x[0], 'score': x[1]/s} for x in rule_sup[k].items()]

    if dataset_using == 'wiki':
        with open('wiki_rule_sup_'+str(rel_idx)+'_relative_order.json', 'w') as f:
            json.dump(rule_sup, f)
    elif dataset_using == 'YAGO':
        with open('YAGO_rule_sup_'+str(rel_idx)+'_relative_order.json', 'w') as f:
            json.dump(rule_sup, f)



def investigate_rule_sup(rel_idx):
    relative_order_len_dict = {2: 1, 3: 3, 4: 6, 5: 10}

    if dataset_using == 'wiki':
        with open('wiki_rule_sup_'+str(rel_idx)+'_relative_order.json','r') as f:
            rule_sup = json.load(f)
    elif dataset_using == 'YAGO':
        with open('YAGO_rule_sup_'+str(rel_idx)+'_relative_order.json','r') as f:
            rule_sup = json.load(f)


    for k in rule_sup.keys():
        for r in rule_sup[k]:
            if len(r['rule'])!= relative_order_len_dict[len(str_to_list(k))//2]:
                print(k)
                print(r['rule'])