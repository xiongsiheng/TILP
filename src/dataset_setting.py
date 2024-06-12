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





def read_dataset_txt(path):
    edges = []
    with open(path, 'r') as f:
        lines=f.readlines()
        for l in lines:
            a=l.strip().split()
            a=[int(x) for x in a]
            b=copy.copy(a)
            b[3] = min(a[3], a[4])
            b[4] = max(a[3], a[4])
            edges.append(b)
    return edges


def obtain_inv_edges(edges, num_rel):
    if isinstance(edges, list):
        edges = np.array(edges)

    edges_ori = edges[edges[:, 1] < (num_rel//2)]
    edges_inv = edges[edges[:, 1] >= (num_rel//2)]
    edges_ori_inv = np.hstack([edges_ori[:, 2:3], edges_ori[:, 1:2] + num_rel//2, edges_ori[:, 0:1], edges_ori[:, 3:]])
    edges_inv_inv = np.hstack([edges_inv[:, 2:3], edges_inv[:, 1:2] - num_rel//2, edges_inv[:, 0:1], edges_inv[:, 3:]])

    edges = np.vstack((edges_ori, edges_inv_inv, edges_ori_inv, edges_inv))

    return edges


def do_normal_setting_for_dataset(dataset_name1, num_rel):
    train_edges = read_dataset_txt('../data/' + dataset_name1 + '/train.txt')
    train_edges = obtain_inv_edges(train_edges, num_rel)

    valid_data = read_dataset_txt('../data/' + dataset_name1 + '/valid.txt')
    valid_data = np.array(valid_data)
    valid_data_inv = np.hstack([valid_data[:, 2:3], valid_data[:, 1:2] + num_rel//2, valid_data[:, 0:1], valid_data[:, 3:]])

    test_data = read_dataset_txt('../data/' + dataset_name1 + '/test.txt')
    test_data = np.array(test_data)
    test_data_inv = np.hstack([test_data[:, 2:3], test_data[:, 1:2] + num_rel//2, test_data[:, 0:1], test_data[:, 3:]])

    return train_edges, valid_data, valid_data_inv, test_data, test_data_inv


def split_dataset(dataset_using, dataset_name1, train_edges, num_rel, num_sample_per_rel=-1):
    pos_examples_idx = []
    bg_train = []
    bg_pred = []

    path = '../data/' + dataset_name1 + '/pos_examples_idx.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            pos_examples_idx = json.load(f)
    # else:
    #     print('pos_examples_idx.json does not exist')

    path = '../data/' + dataset_name1 + '/bg_train.txt'
    if os.path.exists(path):
        bg_train = read_dataset_txt(path)
        bg_train = obtain_inv_edges(bg_train, num_rel)
    # else:
    #     print('bg_train.txt does not exist')

    path = '../data/' + dataset_name1 + '/bg_pred.txt'
    if os.path.exists(path):
        bg_pred = read_dataset_txt(path)
        bg_pred = obtain_inv_edges(bg_pred, num_rel)
    # else:
    #     print('bg_pred.txt does not exist')

   
    pos_examples_idx = np.array(list(range(len(train_edges)))) if len(pos_examples_idx) == 0 else pos_examples_idx

    if num_sample_per_rel > 0:
        pos_examples_idx_sample = []
        for rel_idx in range(num_rel):
            cur_pos_examples_idx = pos_examples_idx[train_edges[:, 1] == rel_idx]
            np.random.shuffle(cur_pos_examples_idx)
            pos_examples_idx_sample.append(cur_pos_examples_idx[:num_sample_per_rel])

        pos_examples_idx_sample = np.hstack(pos_examples_idx_sample)
        pos_examples_idx = pos_examples_idx_sample

    pos_examples_idx = pos_examples_idx.tolist()

    if len(bg_train) == 0:
        bg_train = train_edges.copy()
    if len(bg_pred) == 0:
        bg_pred = train_edges.copy()

    return pos_examples_idx, bg_train, bg_pred


def obtain_total_dataset(dataset_name1):
    edges1 = read_dataset_txt('../data/' + dataset_name1 + '/train.txt')
    edges2 = read_dataset_txt('../data/' + dataset_name1 + '/valid.txt')
    edges3 = read_dataset_txt('../data/' + dataset_name1 + '/test.txt')

    edges = np.vstack((edges1, edges2, edges3))
    return edges

def do_temporal_shift_to_dataset(dataset_name1, num_rel, time_shift_mode=1):
    edges1 = read_dataset_txt('../data/' + dataset_name1 + '/train.txt')
    edges2 = read_dataset_txt('../data/' + dataset_name1 + '/valid.txt')
    edges3 = read_dataset_txt('../data/' + dataset_name1 + '/test.txt')

    edges = np.vstack((edges1, edges2, edges3))
    idx_ls = np.array(range(len(edges))).reshape((-1,1))
    edges = np.hstack((edges, idx_ls))

    if time_shift_mode in [0,1]:
        edges = edges[np.argsort(edges[:, 3])]
    elif time_shift_mode == -1:
        edges = edges[np.argsort(edges[:, 3])][::-1]

    # correct wrong timestamp
    edges[:,3][edges[:,3]>2022] = 2022
    edges[:,4][edges[:,4]>2022] = 2022

    if time_shift_mode == 0:
        anchor = int(len(edges)*0.5)
        train_edges = np.vstack([edges[:anchor-len(edges2)], edges[anchor+len(edges3):]])
        valid_data = edges[anchor-len(edges2):anchor]
        test_data = edges[anchor:anchor + len(edges3)]
    else:
        train_edges = edges[:len(edges1)]
        valid_data = edges[len(edges1):len(edges1)+len(edges2)]
        test_data = edges[len(edges1)+len(edges2):]

    train_idx = train_edges[:,-1].tolist()
    valid_idx = valid_data[:,-1].tolist()
    test_idx = test_data[:,-1].tolist()
    train_edges = train_edges[:,:-1]
    valid_data = valid_data[:,:-1]
    test_data = test_data[:,:-1]

    if 1:
        with open('time_shift_idx.json','w') as f:
            json.dump({'train_idx':train_idx, 'valid_idx':valid_idx, 'test_idx':test_idx}, f)

    train_edges = obtain_inv_edges(train_edges, num_rel)
    valid_data_inv = np.hstack([valid_data[:, 2:3], valid_data[:, 1:2] + num_rel//2, valid_data[:, 0:1], valid_data[:, 3:]])
    test_data_inv = np.hstack([test_data[:, 2:3], test_data[:, 1:2] + num_rel//2, test_data[:, 0:1], test_data[:, 3:]])
    whole_data =  obtain_inv_edges(edges, num_rel)
    return train_edges, valid_data, valid_data_inv, test_data, test_data_inv, whole_data


def make_subset(num_edges, r, name, dataset_name1):
    a = list(range(num_edges))
    np.random.shuffle(a)
    a = a[:int(num_edges * r)]
    with open('../data/' + dataset_name1 + '/' + name + '.json', 'w') as f:
        json.dump(a, f)


def my_get_idx_sub(f_name, dataset_name1, train_edges):
    with open('../data/' + dataset_name1 + '/'+ f_name + '.json') as f:
        train_idx_sub = json.load(f)
        train_idx_sub = train_idx_sub + [idx + len(train_edges)//2 for idx in train_idx_sub]

        train_facts = list(set(range(len(train_edges))) - set(train_idx_sub))

    return train_idx_sub, train_facts


def change_ent_int_based_on_train_edges():
    for k in ent_prop_dict.keys():
        if ent_prop_dict[k] == 'p':
            z1 = train_edges[((train_edges[:,0]==int(k)) | (train_edges[:,2]==int(k))) & (train_edges[:,4]<1000)]
            z2 = train_edges[((train_edges[:,0]==int(k)) | (train_edges[:,2]==int(k))) & (train_edges[:,4]>=1000)]

            z = copy.copy(ent_int_dict[k])
            ent_int_dict[k][0] = int(min([ent_int_dict[k][0]] + [z11[3] for z11 in z1] + [z21[3] for z21 in z2 if z21[3]>1000]))
            ent_int_dict[k][1] = int(max([ent_int_dict[k][1]] + [z11[4] for z11 in z1] + [z21[4] for z21 in z2]))

            if not ent_int_dict[k] == z:
                print(k, z, ent_int_dict[k])



def obtain_assiting_data(dataset_using, dataset_name1, train_edges, num_rel, num_entites):
    outputs = []
    if dataset_using == 'wiki':
        with open('../data/' + dataset_name1 + '/entity_int_dict.json', 'r') as f:
            ent_int_dict = json.load(f)
        for k in ent_int_dict:
            ent_int_dict[k] = [ent_int_dict[k][0], ent_int_dict[k][1]]

        ent_int_mat = np.zeros((num_entites, 2))
        ent_int_valid_mat = np.zeros((num_entites, 2))
        for k in ent_int_dict.keys():
            if isinstance(ent_int_dict[k][0], int):
                ent_int_mat[int(k), 0] = ent_int_dict[k][0]
                ent_int_valid_mat[int(k), 0] = 1
            if isinstance(ent_int_dict[k][1], int):
                ent_int_mat[int(k), 1] = ent_int_dict[k][1]
                ent_int_valid_mat[int(k), 1] = 1

        outputs.append(ent_int_mat)
        outputs.append(ent_int_valid_mat)


        with open('../data/' + dataset_name1 + '/entity2id.json', 'r') as f:
            ent_id_dict1 = json.load(f)
            ent_id_dict = {}
            for k in ent_id_dict1.keys():
                ent_id_dict[str(ent_id_dict1[k])] = k
            ent_id_dict1 = {}
        outputs.append(ent_id_dict)


        rel_int_dict = {}
        for k in ent_int_dict.keys():
            for edge in train_edges[train_edges[:, 2]==int(k)].tolist():
                if edge[1] not in rel_int_dict.keys():
                    rel_int_dict[edge[1]] = []
                rel_int_dict[edge[1]].append(edge[3] - ent_int_dict[k][0])
        outputs.append(rel_int_dict)


        Gauss_int_dict = {}
        for k in rel_int_dict.keys():
            k1 = int(k)
            if k1 not in [40, 17] and len(rel_int_dict[k])>10:
                y = np.array(rel_int_dict[k])
                mu =np.mean(y)
                sigma = max(0.1, np.std(y))
                Gauss_int_dict[k1] = [mu, sigma]

        outputs.append(Gauss_int_dict)

        def create_recur_dict():
            x = np.unique(train_edges[:, :3], axis = 0)
            recur_dict = {}
            for x1 in x:
                y = train_edges[np.all(train_edges[:, :3]==x1, axis=1)]
                if len(y)>1:
                    y = y[y[:, 3].argsort()]
                    y_c = copy.copy(y)
                    y = y[1:, 3] - y[:-1, 4]
                    if x1[1] not in recur_dict.keys():
                        recur_dict[int(x1[1])] = []
                    recur_dict[int(x1[1])] += [int(y1) for y1 in y]
                    if min(y)<0:
                        print(y_c)
            print(recur_dict)

            with open('../data/'+ dataset_name1 +'/wiki_recur_dict.json', 'w') as f:
                json.dump(recur_dict, f)


        def create_recur_int_dict():
            with open('../data/'+ dataset_name1 +'/wiki_recur_dict.json', 'r') as f:
                recur_dict = json.load(f)

            Gauss_recur_int_dict = {}
            for k in recur_dict.keys():
                k1 = int(k)
                recur_dict[k] = [num for num in recur_dict[k] if num>=0]
                if len(recur_dict[k])>5:
                    y = np.array(recur_dict[k])
                    mu =np.mean(y)
                    sigma =np.std(y)
                    Gauss_recur_int_dict[k1] = [mu, sigma]

            return Gauss_recur_int_dict, recur_dict


    elif dataset_using == 'YAGO':
        with open('../data/' + dataset_name1 + '/YAGO_ent_int_new.json', 'r') as f:
            ent_int_dict = json.load(f)

        for k in ent_int_dict:
            ent_int_dict[k] = [ent_int_dict[k][0], ent_int_dict[k][1]]

        ent_int_mat = np.zeros((num_entites, 2))
        ent_int_valid_mat = np.zeros((num_entites, 2))
        for k in ent_int_dict.keys():
            if isinstance(ent_int_dict[k][0], int):
                ent_int_mat[int(k), 0] = ent_int_dict[k][0]
                ent_int_valid_mat[int(k), 0] = 1
            if isinstance(ent_int_dict[k][1], int):
                ent_int_mat[int(k), 1] = ent_int_dict[k][1]
                ent_int_valid_mat[int(k), 1] = 1

        outputs.append(ent_int_mat)
        outputs.append(ent_int_valid_mat)


        with open('../data/' + dataset_name1 + '/ent_prop_dict.json', 'r') as f:
            ent_prop_dict = json.load(f)


        ent_prop_mat = np.zeros((num_entites, 1))
        for k in ent_prop_dict.keys():
            if ent_prop_dict[k] == 'p':
                ent_prop_mat[int(k),0] = 1
            elif ent_prop_dict[k] == 'n':
                ent_prop_mat[int(k),0] = -1

        outputs.append(ent_prop_mat)

        with open('../data/' + dataset_name1 + '/ent_id_dict.json', 'r') as f:
            ent_id_dict = json.load(f)
        outputs.append(ent_id_dict)


        with open('../data/' + dataset_name1 + '/rel_id_dict.json', 'r') as f:
            rel_id_dict = json.load(f)
            tmp = copy.copy(rel_id_dict)
            for k in tmp.keys():
                rel_id_dict[str(int(k) + num_rel//2)] = tmp[k] + '^-1'
        outputs.append(rel_id_dict)


        query_prop_dict = {'p2p': [4, 14], 'p2n': [0, 7, 1, 2, 3, 6, 8], 'u2n': [5], 'n2p': [10, 17, 11, 12, 13, 16, 18]}
        outputs.append(query_prop_dict)


        rel_int_dict = {}
        for k in ent_int_dict.keys():
            for edge in train_edges[train_edges[:, 2]==int(k)].tolist():
                if not (edge[3]<1000 and edge[4]>1000):
                    if edge[1] not in rel_int_dict.keys():
                        rel_int_dict[edge[1]] = []
                    rel_int_dict[edge[1]].append(edge[3] - ent_int_dict[k][0])
        outputs.append(rel_int_dict)


        Gauss_int_dict = {}
        for k in rel_int_dict.keys():
            if int(k) not in [5, 9, 15, 19]:
                y = np.array(rel_int_dict[k])
                mu =np.mean(y)
                sigma = max(0.1, np.std(y))
                Gauss_int_dict[int(k)] = [mu, sigma]
        outputs.append(Gauss_int_dict)

    if dataset_using == 'wiki':
        assiting_data = [ent_int_mat, ent_int_valid_mat, Gauss_int_dict]
    elif dataset_using == 'YAGO':
        assiting_data = [ent_int_mat, ent_int_valid_mat, Gauss_int_dict, query_prop_dict, ent_prop_mat]
    return assiting_data



def obtain_distribution_parameters(train_edges, num_rel):
    p_rec = np.zeros((num_rel,))
    p_order = -1 * np.ones((num_rel, num_rel))
    mu_pair = -1 * np.ones((num_rel, num_rel))
    sigma_pair = -1 * np.ones((num_rel, num_rel))
    lambda_pair = -1 * np.ones((num_rel, num_rel))
    mu_dur = np.zeros((num_rel,))
    sigma_dur = np.zeros((num_rel,))

    t_s_dict = {}
    for i in range(num_rel):
        for j in range(num_rel):
            t_s_dict[(i,j)] = []

    t_d_dict = {}
    for i in range(num_rel):
        t_d_dict[i] = []

    for i in range(len(train_edges)):
        x = np.delete(train_edges, i, 0)
        x = x[x[:,0] == train_edges[i][0]]
        p_rec[train_edges[i][1]] += 1.0
        t_d_dict[train_edges[i][1]].append(train_edges[i][4]-train_edges[i][3])

        for rel in np.unique(x[:,1]):
            x1 = x[x[:,1] == rel][:,3] - train_edges[i][3]
            idx = np.argmin(np.abs(x1))
            t_s_dict[(train_edges[i][1],rel)].append(x1[idx])

    for i in range(num_rel):
        x = np.array(t_s_dict[(i,i)])
        p_rec[i] = float(len(x))/p_rec[i]

        x = np.array(t_d_dict[i])
        mu_dur[i] = np.mean(np.abs(x))
        sigma_dur[i] = max(0.1, np.std(np.abs(x)))


    for i in range(num_rel):
        for j in range(num_rel):
            x = np.array(t_s_dict[(i,j)])
            if len(x)>0:
                p_order[i,j] = float(len(x[x>=0]))/float(len(x))
                mu_pair[i,j] = np.mean(np.abs(x))
                sigma_pair[i,j] = max(0.1, np.std(np.abs(x)))
                if np.mean(np.abs(x)) == 0:
                    lambda_pair[i,j] = 100
                else:
                    lambda_pair[i,j] = 1.0/np.mean(np.abs(x))

    return p_rec, p_order, mu_pair, sigma_pair, lambda_pair, mu_dur, sigma_dur


def obtain_distribution_parameters_Wc(train_edges, dataset_using, overall_mode, num_rel):
    p_order = -1 * np.ones((num_rel, num_rel))
    mu_pair = -1 * np.ones((num_rel, num_rel))
    sigma_pair = -1 * np.ones((num_rel, num_rel))
    lambda_pair = -1 * np.ones((num_rel, num_rel))

    t_s_dict = {}
    for i in range(num_rel):
        for j in range(num_rel):
            t_s_dict[(i,j)] = []

    for i in range(len(train_edges)):
        if overall_mode == 'general':
            path = '../output/found_time_gaps/'+ dataset_using +'_train_query_'+str(i)+ '.json'
        elif overall_mode in ['few', 'biased', 'time_shifting']:
            path = '../output/found_time_gaps_'+ overall_mode +'/'+ dataset_using +'_train_query_'+str(i)+ '.json'

        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

            for k in data.keys():
                if isinstance(data[k], int):
                    t_s_dict[(train_edges[i][1], int(k))].append(data[k])


    for i in range(num_rel):
        for j in range(num_rel):
            x = np.array(t_s_dict[(i,j)])
            if len(x)>0:
                p_order[i,j] = float(len(x[x>=0]))/float(len(x))
                mu_pair[i,j] = np.mean(np.abs(x))
                sigma_pair[i,j] = max(0.1, np.std(np.abs(x)))
                if np.mean(np.abs(x)) == 0:
                    lambda_pair[i,j] = 100
                else:
                    lambda_pair[i,j] = 1.0/np.mean(np.abs(x))

    return p_order, mu_pair, sigma_pair, lambda_pair