import json
from joblib import Parallel, delayed
import os
from tqdm import tqdm





def my_predict(my_model, i, num_queries, num_processes, rel_idx, bg_pred, test_data, test_data_inv,
                const_pattern_ls, assiting_data, dist_pars, train_edges):
    rank_dict = my_model.predict_in_batch(i, num_queries, num_processes, rel_idx, bg_pred, 
                                            test_data, test_data_inv,
                                            const_pattern_ls, assiting_data, dist_pars, train_edges)
    return rank_dict



def do_my_predict(rel_ls, my_model, dataset_using, bg_pred, test_data, test_data_inv, 
                    const_pattern_ls, assiting_data, dist_pars, train_edges, num_processes = 24):
    for this_rel in tqdm(rel_ls, desc='predicting: '):
        num_queries = len(test_data) // num_processes
        output = Parallel(n_jobs=num_processes)(
            delayed(my_predict)(my_model, i, num_queries, num_processes, this_rel, bg_pred, 
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


# def obtain_id_dict():
#     if dataset_using == 'YAGO':
#         with open('../data/YAGO11k/entity2id.txt') as f:
#             lines = f.readlines()
#     elif dataset_using == 'wiki':
#         with open('../data/WIKIDATA12k/entity2id.txt') as f:
#             lines = f.readlines()


#     id_dict = {}
#     for j in range(len(lines)):
#         line = lines[j]
#         z, id1 = line.strip().split('\t')[:2]
#         if dataset_using == 'YAGO':
#             id_dict[id1] = z[1:-1]
#         elif dataset_using == 'wiki':
#             id_dict[id1] = z

#     return id_dict


# def explore_ent_int_dict():
#     id_dict = obtain_id_dict()
#     cnt = 0
#     for j in range(len(test_data)):
#         line = test_data[j]
#         for k in [line[0], line[2]]:
#             if str(k) in ent_int_dict.keys():
#                 if isinstance(ent_int_dict[str(k)][0], int):
#                     if not (ent_int_dict[str(k)][0]<=line[3]):
#                         print('err', j)
#                         print(k, id_dict[str(k)], ent_int_dict[str(k)])
#                         print(line)
#                         cnt += 1
#                         continue
#                 if isinstance(ent_int_dict[str(k)][1], int):
#                     if not (ent_int_dict[str(k)][1]>=line[4]):
#                         if dataset_using == 'wiki':
#                             if line[1] in [17, 20]:
#                                 continue
#                         print('err', j)
#                         print(k, id_dict[str(k)], ent_int_dict[str(k)])
#                         print(line)
#                         cnt += 1
#                         continue
#     print(cnt)


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