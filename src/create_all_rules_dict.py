import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from Models import TILP


def my_create_all_rule_dicts(i, n_s, n_p, rel_idx, train_edges, model_paras, num_rel):
    n_t = len(train_edges)//2
    n_r = n_t - (i + 1) * n_p
    s = 0
    if rel_idx >= num_rel//2:
        s = len(train_edges)//2

    if (n_r >= n_s) and (i + 1 < n_p):
        idxs = range(s+i * n_s, s+(i + 1) * n_s)
    else:
        idxs = range(s+i * n_s, s+n_t)

    my_model = TILP(*model_paras)
    rule_dict, rule_sup_num_dict = my_model.create_all_rule_dicts(rel_idx, train_edges, idxs)
    
    return rule_dict, rule_sup_num_dict


def do_my_create_all_rule_dicts(this_rel, train_edges, model_paras, num_rel, num_processes=24):
    n_s = (len(train_edges)//2) // num_processes
    output = Parallel(n_jobs=num_processes)(
        delayed(my_create_all_rule_dicts)(i, n_s, num_processes, this_rel, train_edges, 
                                            model_paras, num_rel) 
                                            for i in range(num_processes)
    )
    my_model = TILP(*model_paras)
    rule_sup_num_dict, rule_dict = {}, {}
    for i in range(num_processes):
        rule_sup_num_dict = my_model.my_merge_dict(rule_sup_num_dict, output[i][1])
        for k in output[i][0].keys():
            if k not in rule_dict.keys():
                rule_dict[k] = output[i][0][k].copy()
            else:
                rule_dict[k] = np.vstack((rule_dict[k], output[i][0][k]))
            rule_dict[k] = np.unique(rule_dict[k], axis=0)

    # for k in rule_dict.keys():
    #     print('ruleLen', str(k), 'ruleShape', rule_dict[k].shape)

    return rule_dict, rule_sup_num_dict


def do_rule_summary(rel_ls, model_paras, num_rel, train_edges, const_pattern_ls, num_processes=24):
    my_model = TILP(*model_paras)
    for rel in rel_ls:
        # collect data
        rule_dict, rule_sup_num_dict = do_my_create_all_rule_dicts(rel, train_edges, model_paras, num_rel, num_processes=num_processes)
        # store the rules
        my_model.write_all_rule_dicts(rel, rule_dict, rule_sup_num_dict, const_pattern_ls, cal_score=False, select_topK_rules=True)

    return

        


def do_calculate_rule_scores(rel_ls, model_paras, num_rel, train_edges, const_pattern_ls, num_processes=24):
    my_model = TILP(*model_paras)
    for rel in tqdm(rel_ls, desc='rule summary: '):
        # collect data
        rule_dict, rule_sup_num_dict = do_my_create_all_rule_dicts(rel, train_edges, model_paras, num_rel, num_processes=num_processes)
        # calculate score
        my_model.write_all_rule_dicts(rel, rule_dict, rule_sup_num_dict, const_pattern_ls, cal_score=True, select_topK_rules=False)

    return



# def explore_all_rules_dict(rel_idx):
#     with open('output/learned_rules/' + dataset_using +'_all_rules_'+str(rel_idx)+'.json','r') as f:
#         rule_dict1 = json.load(f)

#     for k in rule_dict1.keys():
#         print(k, len(rule_dict1[k]))
#         # x = [r['score'] for r in rule_dict1[k] if r['score']>1e-30]
#         # print(min(x), max(x), len(x))
#         x = np.array([r['rule'] for r in rule_dict1[k]])
#         print(x.shape)
#         y = np.unique(x, axis=0)
#         print(y.shape)
#         y = np.unique(x[:,:int(k)], axis=0)
#         print(y.shape)
#         y = np.unique(x[:,:int(k)-1], axis=0)
#         print(y.shape)
#         print(' ')
    
#     return