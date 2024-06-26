import json
from joblib import Parallel, delayed
import os
from tqdm import tqdm
from Models import *




def my_predict(model_paras, i, num_queries, num_processes, rel_idx, bg_pred, test_data, test_data_inv,
                const_pattern_ls, assiting_data, dist_pars, train_edges):
    my_model = Predictor(*model_paras)
    rank_dict = my_model.predict_in_batch(i, num_queries, num_processes, rel_idx, bg_pred, 
                                            test_data, test_data_inv,
                                            const_pattern_ls, assiting_data, dist_pars, train_edges)
    return rank_dict



def do_my_predict(rel_ls, model_paras, dataset_using, bg_pred, test_data, test_data_inv, 
                    const_pattern_ls, assiting_data, dist_pars, train_edges, num_processes = 24):
    for this_rel in tqdm(rel_ls, desc='predicting: '):
        num_queries = len(test_data) // num_processes
        output = Parallel(n_jobs=num_processes)(
            delayed(my_predict)(model_paras, i, num_queries, num_processes, this_rel, bg_pred, 
                                test_data, test_data_inv, const_pattern_ls, assiting_data, dist_pars, train_edges) 
                                for i in range(num_processes)
        )

        rank_dict = {}
        for i in range(num_processes):
            rank_dict.update(output[i])

        path = '../output/rank_dict/'+ dataset_using +'_rank_dict_'+str(this_rel)+'.json'
        with open(path, 'w') as f:
            json.dump(rank_dict, f)



def evaluate_ranks(ranks):
    num_samples = len(ranks)
    if num_samples==0:
        return {'hits_1': None, 'hits_3': None, 'hits_10': None, 'mrr': None}

    hits_1, hits_3, hits_10, mrr = 0., 0., 0., 0.

    if isinstance(ranks, dict):
        k_ls = ranks.keys()
    elif isinstance(ranks, list):
        k_ls = range(len(ranks))

    for k in k_ls:
        if isinstance(ranks[k], int) or isinstance(ranks[k], float):
            rank = ranks[k]

            if rank <= 10:
                hits_10 += 1.
                if rank <= 3:
                    hits_3 += 1.
                    if rank == 1:
                        hits_1 += 1.
            mrr += 1. / rank

    hits_1 /= num_samples
    hits_3 /= num_samples
    hits_10 /= num_samples
    mrr /= num_samples

    return {'hits_1': hits_1, 'hits_3': hits_3, 'hits_10': hits_10, 'mrr': mrr}



def do_evaluate(rel_ls, dataset_using):
    all_ranks = {}
    res_dict = {}
    for i in rel_ls:
        path = '../output/rank_dict/'+ dataset_using + '_rank_dict_' + str(i) +'.json'

        if not os.path.exists(path):
            # print(path + ' not found')
            continue
        with open(path,'r') as f1:
            x = json.load(f1)

        all_ranks.update(x)
        res_dict[i] = evaluate_ranks(x)
        print('Relation ' + str(i) + ':')
        print(res_dict[i])

    res_dict['total'] = evaluate_ranks(all_ranks)
    res_dict['total']['num_samples'] = len(all_ranks)
    print('Total: num_samples = ' + str(len(all_ranks)))
    print(evaluate_ranks(all_ranks))
    return res_dict