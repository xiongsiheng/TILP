import json
from joblib import Parallel, delayed
from tqdm import tqdm





def my_apply(i, num_queries, num_processes, rel_idx, mode, train_edges, my_model, 
                path_name='', pos_examples_idx=None, time_shift_mode=0):
    my_model.apply_in_batch(i, num_queries, num_processes, rel_idx, mode, train_edges, 
                                    path_name, pos_examples_idx, time_shift_mode)



def do_my_find_rules(rel_ls, train_edges, my_model, mode='path_search', path_name='', 
                    pos_examples_idx=None, time_shift_mode=0, num_processes=24):
    for this_rel in tqdm(rel_ls, desc=mode):
        num_queries = (len(train_edges)//2) // num_processes
        Parallel(n_jobs=num_processes)(
            delayed(my_apply)(i, num_queries, num_processes, this_rel, mode, train_edges, \
                            my_model, path_name, pos_examples_idx, time_shift_mode) for i in range(num_processes)
            )



def check_application_results(train_edges, dataset_using):
    for idx in range(len(train_edges)//2):
        with open('output/found_rules/' + dataset_using +'_train_query_'+str(idx)+'.json') as f:
            rule_dict = json.load(f)
        with open('output/found_rules/' + dataset_using +'_train_query_'+str(idx + len(train_edges)//2)+'.json') as f:
            rule_dict_inv = json.load(f)

        for k in rule_dict.keys():
            if not len(rule_dict[k]) == len(rule_dict_inv[k]):
                print(idx)
                break



def check_certain_application_results(idx_ls):
    for idx in idx_ls:
        with open('output/found_rules/YAGO_train_query_'+str(idx)+'.json') as f:
            rule_dict = json.load(f)

        for k in rule_dict.keys():
            print(k, len(rule_dict[k]))