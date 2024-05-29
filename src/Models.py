import random
import numpy as np
import sys
import json
import os
import pandas as pd
import copy
from collections import Counter
from scipy.stats import norm

import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict

from utlis import gadgets




class Walker(gadgets):
    def __init__(self, num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode):
        super(Walker, self).__init__(num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode)

    def create_adj_mat(self, masked_facts):
        edges_simp = masked_facts[:,[0,2]]
        edges_simp = np.unique(edges_simp, axis=0)
        edges_simp = edges_simp.astype(int)
        pos = list(edges_simp)
        rows, cols = zip(*pos)

        adj_mat = np.zeros((self.num_entites, self.num_entites))
        adj_mat[rows, cols] = 1

        return adj_mat

    def BFS_mat_ver2(self, st_node, adj_mat, num_nodes, targ_node, max_len):
        node_st = np.zeros((num_nodes, 1))
        node_st[int(st_node)] = 1
        res = node_st.copy()


        new_nodes_ls =[[int(st_node)]]
        num_hops = []
        for i in range(max_len):
            res = np.dot(adj_mat, res)
            res[res>1] = 1

            idx_ls = np.where(res==1)[0]

            # cur_new_idx_ls = list(set(idx_ls)-set(idx_ls_old))
            cur_new_idx_ls = idx_ls.copy()
            if len(cur_new_idx_ls) > 0:
                new_nodes_ls.append(cur_new_idx_ls)

            if res[int(targ_node)] == 1:
                num_hops.append(i+1)
                # res[targ_node] = 0

        return num_hops, new_nodes_ls

    def get_walks(self, walk_edges, columns, rule=None, rule_Len=0):
        df_edges = []
        df = pd.DataFrame(
            walk_edges[0],
            columns=[c + str(0) for c in columns] + ["entity_" + str(1)],
            dtype=int,
        )

        df_edges.append(df)
        df = df[0:0]

        for i in range(1, len(walk_edges)):
            df = pd.DataFrame(
                walk_edges[i],
                columns=[c + str(i) for c in columns] + ["entity_" + str(i+1)],
                dtype=int,
            )

            df_edges.append(df)
            df = df[0:0]

        rule_walks = df_edges[0]
        df_edges[0] = df_edges[0][0:0]

        idx_TR_ls = 0
        for i in range(1, len(df_edges)):
            rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
            rule_walks = self.remove_matching_rows(rule_walks) # remove path that contains repeated edges 
            if (rule is not None) and (self.f_non_Markovian):
                if self.f_adjacent_TR_only:
                    # remove path that does not satisfy the TR constraint
                    rule_walks_np = rule_walks.to_numpy()
                    idx_st = 4*(i-1) + 2 # [s, r, t_s, t_e, o]
                    idx_ed = 4*i + 2
                    # # since we remove previous time information
                    # idx_st -= 2*(i-1)
                    # idx_ed -= 2*(i-1)
                    cur_TR = self.obtain_tmp_rel(rule_walks_np[:, idx_st:idx_st+2], rule_walks_np[:, idx_ed:idx_ed+2])
                    mask = pd.Series(cur_TR == rule[rule_Len+i+1])
                    rule_walks = rule_walks[mask]

                    # remove previous time information (we cannot since we need to remove path that contains repeated edges)
                    # preserve_list_row = [1] * i + [0, 0, 1, 1, 1, 1]
                    # if i == len(df_edges)-1:
                    #     preserve_list_row = [1] * i + [0, 0, 1, 0, 0, 1] # remove the last time information
                    # mask_row = pd.Series(preserve_list_row, index=rule_walks.columns).astype(bool)
                    # rule_walks = rule_walks.loc[:, mask_row]
                else:
                    # remove path that does not satisfy the TR constraint
                    for j in range(i):
                        idx_st, idx_ed = 4*j + 2, 4*i + 2
                        rule_walks_np = rule_walks.to_numpy()
                        cur_TR = self.obtain_tmp_rel(rule_walks_np[:, idx_st:idx_st+2], rule_walks_np[:, idx_ed:idx_ed+2])
                        mask = pd.Series(cur_TR == rule[rule_Len*2 + idx_TR_ls])
                        idx_TR_ls += 1
                        rule_walks = rule_walks[mask].reset_index(drop=True)
                        

            df_edges[i] = df_edges[i][0:0]
        
        return rule_walks


    def path_search(self, masked_facts, query, time_shift_mode):
        '''
        Different TR setting:
        f_Markovian: TR(I_1, I_q), TR(I_2, I_q), ..., TR(I_N, I_q)
        f_non_Markovian: TR(I_1, I_q), TR(I_2, I_q), ..., TR(I_N, I_q), TR(I_1, I_2), TR(I_1, I_3), TR(I_2, I_3), TR(I_1, I_4), ..., TR(I_{N-1}, I_N)
        f_non_Markovian and f_adjacent_TR_only: TR(I_1, I_q), TR(I_N, I_q), TR(I_1, I_2), TR(I_2, I_3), ..., TR(I_{N-1}, I_N)
        '''
        rule_dict = {}

        adj_mat = self.create_adj_mat(masked_facts)
        cur_num_hops1, new_nodes_ls1 = self.BFS_mat_ver2(query[0], adj_mat, self.num_entites, query[2], self.max_explore_len)
        
        if len(cur_num_hops1) > 0:
            _, new_nodes_ls2 = self.BFS_mat_ver2(query[2], adj_mat, self.num_entites, query[0], self.max_explore_len)
            for num in cur_num_hops1:
                path_ls = self.find_common_nodes(new_nodes_ls1[:num+1], new_nodes_ls2[:num+1][::-1])
                walk_edges = []
                for i in range(num):
                    related_facts = masked_facts[np.isin(masked_facts[:,0], path_ls[i]) & np.isin(masked_facts[:,2], path_ls[i+1])]
                    walk_edges.append(related_facts[:,[0,1,3,4,2]]) # [s, r, ts, te, o]

                cur_ent_walk_res = self.get_walks(walk_edges, ["entity_" , "rel_", "ts_", "te_"]).to_numpy()
                # print(cur_ent_walk_res)

                if len(cur_ent_walk_res) == 0:
                    continue

                rules = cur_ent_walk_res[:, [4*i+1 for i in range(num)]] # extarct all r in the path

                for i in range(num):
                    # TR(I_i, I_q)
                    if self.f_non_Markovian and self.f_adjacent_TR_only and i != 0 and i != num-1:
                        # instead of considering all TR, we only preserve adjacent TR: TR(I_1, I_q), TR(I_N, I_q)
                        continue
                    cur_TR = self.obtain_tmp_rel(cur_ent_walk_res[:, 4*i+2:4*i+4], query[3:]).reshape((-1,1))
                    rules = np.hstack((rules, cur_TR))

                if self.f_non_Markovian:
                    # TR(I_j, I_i)
                    for i in range(1, num):
                        for j in range(i):
                            if self.f_adjacent_TR_only and j != i - 1:
                                # instead of considering all TR, we preserve adjacent TR: TR(I_{i-1}, I_i)
                                continue
                            cur_TR = self.obtain_tmp_rel(cur_ent_walk_res[:, 4*j+2:4*j+4], cur_ent_walk_res[:, 4*i+2:4*i+4]).reshape((-1,1))
                            rules = np.hstack((rules, cur_TR))
                
                rules = np.unique(rules, axis=0).tolist()
                # print(rules)

                for r in rules:
                    if time_shift_mode in [-1, 1]:
                        if self.f_adjacent_TR_only and time_shift_mode in r[num:num + 1 + int(num>1)]: # we only consider adjacent TR
                            continue
                        if (not self.f_adjacent_TR_only) and time_shift_mode in r[num:num*2]:
                            continue                                             
                    rule_dict[num] = [] if num not in rule_dict else rule_dict[num]
                    rule_dict[num].append({'rule': r, 'alpha': 'Unknown'})

        return rule_dict


    def calculate_ts_dist(self, path2, masked_facts, query, time_shift_mode):        
        # Todo: consider time shifting setting
        t_s_dict = {}
        for k1 in range(self.num_rel):
            t_s_dict[k1] = []

        adj_mat = self.create_adj_mat(masked_facts)
        cur_num_hops1, new_nodes_ls1 = self.BFS_mat_ver2(query[0], adj_mat, self.num_entites, query[2], self.max_explore_len)
        
        if len(cur_num_hops1) > 0:
            _, new_nodes_ls2 = self.BFS_mat_ver2(query[2], adj_mat, self.num_entites, query[0], self.max_explore_len)
            for num in cur_num_hops1:
                path_ls = self.find_common_nodes(new_nodes_ls1[:num+1], new_nodes_ls2[:num+1][::-1])
                for i in range(num):
                    z = masked_facts[np.isin(masked_facts[:,0], path_ls[i]) & np.isin(masked_facts[:,2], path_ls[i+1])]
                    for z1 in z:
                        t_s_dict[z1[1]].append(z1[3]-query[3])

            for k1 in t_s_dict.keys():   
                t_s_dict[k1] = t_s_dict[k1][np.argmin(np.abs(t_s_dict[k1]))] if len(t_s_dict[k1])>0 else None
            with open(path2, 'w') as f:
                json.dump(t_s_dict, f)
        return 
    

    def apply_single_rule(self, start_node, rule, rule_Len, facts, 
                                f_print=0, return_walk=False, targ_node=None, 
                                return_edges=False):
        rel_ls = rule[:rule_Len] # [r_1, r_2, ..., r_N]
        TR_ls = rule[rule_Len:]  # [TR(I_1, I_q), TR(I_N, I_q), TR(I_1, I_2), TR(I_2, I_3) ...] if adjacent TRs only else [TR(I_1, I_q), TR(I_2, I_q), ..., TR(I_N, I_q), TR(I_1, I_2), TR(I_1, I_3) ...]

        path_ls = []
        cur_nodes = [int(start_node)]
        facts = facts.astype(int) # [s, r, o, TR(I, I_q), ts, te]

        for i in range(rule_Len):
            if self.f_non_Markovian and self.f_adjacent_TR_only and i != 0 and i != rule_Len - 1:
                cur_edges = facts[np.isin(facts[:, 0], cur_nodes) & (facts[:, 1] == rel_ls[i])]
            else:
                # if only adjacent TRs are considered
                cur_edges = facts[np.isin(facts[:, 0], cur_nodes) & (facts[:, 1] == rel_ls[i]) & (facts[:, 3] == TR_ls[min(i, 1)])] if self.f_adjacent_TR_only else \
                            facts[np.isin(facts[:, 0], cur_nodes) & (facts[:, 1] == rel_ls[i]) & (facts[:, 3] == TR_ls[i])]
  
            # print(cur_edges.shape)
            # print(cur_edges)
            # print('---------------------')

            if len(cur_edges) == 0:
                if return_edges:
                    return []
                return ({}, []) if return_walk else {}
                
            cur_nodes = cur_edges[:, 2]
            path_ls.append(cur_edges[:,[0,1,4,5,2]])

        if isinstance(targ_node, int) or isinstance(targ_node, float):
            path_ls[-1] = path_ls[-1][path_ls[-1][:,-1] == targ_node]

        if return_edges:
            return path_ls
       
        heads = ['entity_', 'rel_', 'ts_', 'te_'] # we need to remove path that contains repeated edges
        cur_ent_walk_res = self.get_walks(path_ls, heads, rule, rule_Len).to_numpy()
        cur_nodes = cur_ent_walk_res[:,-1:]

        if f_print:
            print(cur_ent_walk_res)

        path_ls = []
        cur_ent_walk_res = 0 if not return_walk else cur_ent_walk_res

        df = pd.DataFrame(cur_nodes, columns=["end_node"], dtype=int)
        res = df["end_node"].value_counts(normalize=True).to_dict()

        return (res, cur_ent_walk_res) if return_walk else res


    def alpha_calculation(self, path1, masked_facts, query, time_shift_mode):
        rule_dict = {}
        if not os.path.exists(path1):
            return rule_dict
        with open(path1, 'r') as f:
            rule_dict_loaded = json.load(f)
        
        if self.overall_mode == 'general':
            with open('../output/learned_rules/'+ self.dataset_using +'_all_rules_'+str(query[1])+'.json','r') as f:
                rule_dict_summarized = json.load(f)
        elif self.overall_mode in ['few', 'biased', 'time_shifting']:
            with open('../output/learned_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_all_rules_'+str(query[1])+'.json','r') as f:
                rule_dict_summarized = json.load(f)

        TR = self.obtain_tmp_rel(masked_facts[:, 3:], query[3:]).reshape((-1,1))
        masked_facts_with_TR = np.hstack((masked_facts[:, :3], TR)) if not self.f_non_Markovian else np.hstack((masked_facts[:, :3], TR, masked_facts[:, 3:]))
        masked_facts_with_TR = np.unique(masked_facts_with_TR, axis=0)                


        for num in rule_dict_loaded:
            rules = [r_dict['rule'] for r_dict in rule_dict_loaded[num]]
            for r in rules:
                if r not in [r_dict['rule'] for r_dict in rule_dict_summarized[num]]:
                    # Instead of calculating alpha for all rules, we select most frequent ones
                    continue
                if time_shift_mode in [-1, 1] and time_shift_mode in r[int(num):int(num) + 2]:
                    continue
                # print(r)
                # we also consider randomly selecting samples for alpha calculation
                rand_num = random.random()
                if rand_num <= self.prob_cal_alpha:
                    cur_dict = self.apply_single_rule(query[0], r, int(num), masked_facts_with_TR)
                    # print(cur_dict)
                    
                    # if we use path sampling, it's possible we cannot find the targ node
                    # alpha = 0.01 
                    # if query[2] in cur_dict.keys():
                    #     alpha = cur_dict[query[2]]
                    
                    alpha = cur_dict[query[2]]
                else:
                    alpha = 'Unknown'

                if num not in rule_dict:
                    rule_dict[num] = []
                rule_dict[num].append({'rule': r, 'alpha': alpha})
        return rule_dict
       

    def apply(self, train_edges, rel_idx=None, idx_ls=None, pos_examples_idx=None, time_shift_mode=0, mode='path_search'):
        '''
        mode: 'path_search' or 'alpha_calculation'
        1) path_search: given a query, find all paths that satisfy the query
        2) alpha_calculation: given a query, calculate alpha for all rules
        '''
        assert mode in ['path_search', 'alpha_calculation']
        assert self.overall_mode in ['general', 'few', 'biased', 'time_shifting']

        idx_ls1 = idx_ls if idx_ls is not None else range(len(train_edges))

        for idx in idx_ls1:
            path1 = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json' if self.overall_mode == 'general' else \
                    '../output/found_rules_'+ self.overall_mode +'/'+ self.dataset_using + '_train_query_'+str(idx)+'.json'
            path2 = '../output/found_t_s/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json' if self.overall_mode == 'general' else \
                    '../output/found_t_s_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json'

            query = train_edges[idx]
            # print(query, idx)

            if isinstance(rel_idx, int) and (query[1] != rel_idx):
                continue

            if (pos_examples_idx is not None) and (idx not in pos_examples_idx):
                continue

            # mask ori query and inv query in the train set
            masked_facts = np.delete(train_edges, [idx, self.get_inv_idx(len(train_edges)//2, idx)], 0)
            masked_facts = masked_facts[masked_facts[:,3]<=query[3]] if self.overall_mode == 'time_shifting' else masked_facts
            
            if mode == 'path_search':
                rule_dict = self.path_search(masked_facts, query, time_shift_mode)
            else:
                rule_dict = self.alpha_calculation(path1, masked_facts, query, time_shift_mode)
            
            with open(path1, 'w') as f:
                json.dump(rule_dict, f)

            if self.f_Wc_ts:
                self.calculate_ts_dist(path2, masked_facts, query, time_shift_mode)

        return 



    def apply_in_batch(self, i, num_queries, num_processes, rel_idx, mode, train_edges, path_name='', 
                            pos_examples_idx=None, time_shift_mode=0):
        queries_idx = self.create_training_idx_in_batch(i, num_queries, num_processes, rel_idx, train_edges)
        self.apply(train_edges, rel_idx, queries_idx, pos_examples_idx, time_shift_mode, mode=mode)

        return




    # def obtain_sp_rule_walks(self, ent_walks, rule, rule_Len, facts):
    #     rel_ls = rule[:rule_Len]
    #     pattern_ls = rule[rule_Len:]

    #     path_ls = []
    #     for i in range(rule_Len):
    #         y = facts[np.isin(facts[:, 0], ent_walks[:,i]) & (facts[:, 1] == rel_ls[i]) & \
    #                             np.isin(facts[:, 2], ent_walks[:,i+1]) & (facts[:, 3] == pattern_ls[i])]
    #         path_ls.append(y[:, [0,4,5,2]])

    #     # print(path_ls)

    #     z = ['entity_', 'ts_', 'te_']
    #     cur_ent_walk_res = self.get_walks(path_ls, z).to_numpy()
    #     path_ls = []

    #     sp_ls = []
    #     for i1 in range(1, rule_Len):
    #         for i2 in range(i1):
    #             # print(i1,i2)
    #             x = self.obtain_tmp_rel(cur_ent_walk_res[:,3*i1+1:3*i1+3], cur_ent_walk_res[:,3*i2+1:3*i2+3])
    #             sp_ls.append(x.reshape(-1,1))

    #     y = cur_ent_walk_res[:, [3*i1 for i1 in range(rule_Len)]]

    #     x = np.hstack([y, np.hstack(sp_ls), cur_ent_walk_res[:,-1:]])
    #     x = np.unique(x, axis=0)
    #     return x[:, rule_Len:]


    # def explore(self, j, train_edges):
    #     line = train_edges[j,:]
    #     masked_facts = np.delete(train_edges, [j, self.get_inv_idx(int(len(train_edges)/2), j)], 0)

    #     edges_simp = masked_facts[:,[0,2]]
    #     edges_simp = np.unique(edges_simp, axis=0)
    #     edges_simp = edges_simp.astype(int)
    #     pos = list(edges_simp)
    #     rows, cols = zip(*pos)

    #     adj_mat = np.zeros((self.num_entites, self.num_entites))
    #     adj_mat[rows, cols] = 1

    #     cur_num_hops1, new_nodes_ls1 = self.BFS_mat_ver2(line[0], adj_mat, self.num_entites, line[2], self.max_explore_len)

    #     ent_walk_res = {}
    #     # print(line)
    #     if len(cur_num_hops1) > 0:
    #         cur_num_hops2, new_nodes_ls2 = self.BFS_mat_ver2(line[2], adj_mat, self.num_entites, line[0], self.max_explore_len)

    #         for num in cur_num_hops1:
    #             path_ls = self.find_common_nodes(new_nodes_ls1[:num+1], new_nodes_ls2[:num+1][::-1])


    #             walk_edges = []
    #             for i in range(num):
    #                 related_facts = masked_facts[np.isin(masked_facts[:,0], path_ls[i]) & np.isin(masked_facts[:,2], path_ls[i+1])
    #                                                       & (masked_facts[:,3]<=line[3])]
    #                 f_x = 0
    #                 if len(related_facts) == 0:
    #                     related_facts = masked_facts[np.isin(masked_facts[:,0], path_ls[i]) & np.isin(masked_facts[:,2], path_ls[i+1])]
    #                     f_x = 1
    #                 z = np.unique(related_facts[:,:3], axis=0)
    #                 y = []
    #                 for edge in z:
    #                     x = np.all(related_facts[:,:3] == edge, axis=1)
    #                     if f_x == 0:
    #                         t_s = max(related_facts[x][:,3])
    #                         t_e = max(related_facts[x & (related_facts[:,3] == t_s)][:,4])
    #                     else:
    #                         t_s = min(related_facts[x][:,3])
    #                         t_e = min(related_facts[x & (related_facts[:,3] == t_s)][:,4])
    #                     y.append([edge[0], edge[1], t_s, t_e, edge[2]])

                    
    #                 related_facts = np.array(y)
    #                 walk_edges.append(related_facts)

    #             cur_ent_walk_res = self.get_walks(walk_edges, ["entity_", "rel_", "ts_", "te_"]).to_numpy()

    #             cur_ent_walk_res = self.check_path_rep_v2(cur_ent_walk_res, num)

    #             if len(cur_ent_walk_res)>0:
    #                 x = []
    #                 for i1 in range(num):
    #                     x.append(cur_ent_walk_res[:, 4*i1+1].reshape((-1,1)))

    #                 for i1 in range(num):
    #                     x.append(self.obtain_tmp_rel(cur_ent_walk_res[:, 4*i1+2: 4*i1+4], line[3:]).reshape((-1,1)))

    #                 for i1 in range(1, num):
    #                     for i2 in range(i1):
    #                         x.append(self.obtain_tmp_rel(cur_ent_walk_res[:, 4*i1+2: 4*i1+4], cur_ent_walk_res[:, 4*i2+2: 4*i2+4]).reshape((-1,1)))

    #                 x = np.hstack(x)
    #                 ent_walk_res[num] = np.unique(x, axis=0).tolist()

    #     return 



    # def explore_in_batch(self, i, num_queries, num_processes, train_edges):
    #     x = len(train_edges)//2
    #     num_rest_queries = x - (i + 1) * num_queries
    #     if (num_rest_queries >= num_queries) and (i + 1 < num_processes):
    #         queries_idx = range(i * num_queries, (i + 1) * num_queries)
    #     else:
    #         queries_idx = range(i * num_queries, x)

    #     for idx in queries_idx:
    #         self.explore(idx, train_edges)

    #     return 



    # def explore_queries_v2(self, j, train_edges, test_data, test_data_using=None):
    #     if not isinstance(test_data_using, np.ndarray):
    #         test_data_using = test_data

    #     line = test_data_using[j,:]
    #     # print(line)

    #     edges_simp = train_edges[:,[0,2]]
    #     edges_simp = np.unique(edges_simp, axis=0)
    #     edges_simp = edges_simp.astype(int)
    #     pos = list(edges_simp)
    #     rows, cols = zip(*pos)

    #     adj_mat = np.zeros((self.num_entites, self.num_entites))
    #     adj_mat[rows, cols] = 1

    #     x = self.obtain_tmp_rel(train_edges[:, 3:], line[3:]).reshape((-1,1))
    #     x = np.hstack((train_edges[:, :3], x))
    #     x = np.unique(x, axis=0)
    #     x = x.astype(int)

    #     cur_num_hops1, new_nodes_ls1 = self.BFS_mat_ver2(line[0], adj_mat, self.num_entites, line[2], self.max_explore_len)

    #     ent_walk_res = {}
    #     if len(cur_num_hops1) > 0:
    #         cur_num_hops2, new_nodes_ls2 = self.BFS_mat_ver2(line[2], adj_mat, self.num_entites, line[0], self.max_explore_len)
    #         for num in cur_num_hops1:
    #             path_ls = self.find_common_nodes(new_nodes_ls1[:num+1], new_nodes_ls2[:num+1][::-1])
    #             walk_edges = []
    #             for i in range(num):
    #                 related_facts = x[np.isin(x[:,0], path_ls[i]) & np.isin(x[:,2], path_ls[i+1])]
    #                 walk_edges.append(related_facts[:,[0,1,3,2]])
    #             cur_ent_walk_res = self.get_walks(walk_edges, ["entity_" , "rel_", "tmpRel_"]).to_numpy()
    #             # print(cur_ent_walk_res)
    #             if len(cur_ent_walk_res)>0:
    #                 y = cur_ent_walk_res[:, [3*i+1 for i in range(num)] + [3*i+2 for i in range(num)]]
    #                 y = np.unique(y, axis=0)
    #                 # print(y)
    #                 ent_walk_res[num] = y.tolist()

    #     return ent_walk_res


    # def check_path_rep_v3(self, ent_walk_res, path_len, rel_ls):
    #     # print(ent_walk_res)
    #     for i in range(path_len-1):
    #         for j in range(i+1, path_len):
    #             if self.get_inv_idx(num_rel//2, rel_ls[i]) == rel_ls[j]:
    #                 ent_walk_res = ent_walk_res[np.invert(
    #                                                     (ent_walk_res[:,3*i] == ent_walk_res[:,3*j+3])\
    #                                                     & (ent_walk_res[:,3*i+3] == ent_walk_res[:,3*j])\
    #                                                     & np.all(ent_walk_res[:,3*i+1:3*i+3] == ent_walk_res[:,3*j+1:3*j+3], axis=1)
    #                                                         )]
    #             if rel_ls[i] == rel_ls[j]:
    #                 ent_walk_res = ent_walk_res[np.invert(np.all(ent_walk_res[:,3*i:3*i+4] == ent_walk_res[:,3*j:3*j+4], axis=1))]

    #     x = []
    #     for i in range(path_len):
    #         x += [3*i+1, 3*i+2]
    #     x += [-1]

    #     ent_walk_res = ent_walk_res[:, x]
    #     # print(ent_walk_res)
    #     return ent_walk_res


    # def apply_specific_rules_v2(self, rule_walks, sp_rules):
    #     res_dict = {}
    #     for sp_r in sp_rules:
    #         x = rule_walks[np.all(rule_walks[:,:-1] == sp_r['rule'], axis=1), -1:]
    #         df = pd.DataFrame(x, columns=["end_node"], dtype=int)
    #         cur_dict = df["end_node"].value_counts(normalize=True).to_dict()
    #         for k in cur_dict.keys():
    #             cur_dict[k] = cur_dict[k] * sp_r['score']
    #         res_dict = self.my_merge_dict(res_dict, cur_dict)

    #     return res_dict






class Trainer(gadgets):
    def __init__(self, num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode):
        super(Trainer, self).__init__(num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode)

    def _random_uniform_unit(self, r, c):
        bound = 6./ np.sqrt(c)
        init_matrix = np.random.uniform(-bound, bound, (r, c))
        init_matrix = np.array(map(lambda row: row / np.linalg.norm(row), init_matrix))
        return init_matrix


    def _build_rnn_input(self):
        self.queries = tf.placeholder(tf.int32, [self.rnn_batch_size, self.num_ruleLen])
        self.rnn_query_embedding_params_ls = []
        rnn_inputs_ls = []
        for i in range(self.num_ruleLen+1):
            query_embedding_params = tf.Variable(self._random_uniform_unit(
                                                          self.num_rel + 1,
                                                          self.rnn_query_embed_size), 
                                                      dtype=tf.float32)
            rnn_inputs_ls.append(tf.nn.embedding_lookup(query_embedding_params, 
                                            self.queries))
            self.rnn_query_embedding_params_ls.append(query_embedding_params)

        return rnn_inputs_ls


    def build_rnn_graph(self):
        rnn_inputs_ls = self._build_rnn_input()

        self.rnn_inputs_ls = [[tf.reshape(q, [-1, self.rnn_query_embed_size]) 
                                for q in tf.split(rnn_inputs, 
                                             self.num_ruleLen, 
                                             axis=1)] for rnn_inputs in rnn_inputs_ls]

        cell = tf.nn.rnn_cell.LSTMCell(self.rnn_state_size, state_is_tuple=True)

        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.rnn_num_layer, state_is_tuple=True)

        init_state = self.cell.zero_state(self.rnn_batch_size, tf.float32)


        self.W_P = tf.Variable(np.random.randn(
                                self.rnn_state_size, 
                                self.num_rel), 
                            dtype=tf.float32)
        self.b_P = tf.Variable(np.zeros(
                                (1, self.num_rel)), 
                            dtype=tf.float32)

        self.W_TR = tf.Variable(np.random.randn(
                                self.rnn_state_size, 
                                self.num_pattern), 
                            dtype=tf.float32)
        self.b_TR = tf.Variable(np.zeros(
                                (1, self.num_pattern)), 
                            dtype=tf.float32)

        self.W_Len = tf.Variable(np.random.randn(
                                self.rnn_state_size, 
                                self.num_ruleLen), 
                            dtype=tf.float32)
        self.b_Len = tf.Variable(np.zeros(
                                (1, self.num_ruleLen)), 
                            dtype=tf.float32)

        if self.f_non_Markovian:
            self.W_TR_prime = tf.Variable(np.random.randn(
                                    2*self.rnn_state_size, 
                                    self.num_pattern), 
                                dtype=tf.float32)
            self.b_TR_prime = tf.Variable(np.zeros(
                                    (1, self.num_pattern)), 
                                dtype=tf.float32)


        attn_ruleLen = tf.nn.softmax(tf.matmul(self.rnn_inputs_ls[0][0], self.W_Len) + self.b_Len)
        attn_rel_ls, attn_TR_ls, attn_TR_prime_ls  = [], [], []

        for i in range(1,self.num_ruleLen+1):
            rnn_outputs, final_state = tf.contrib.rnn.static_rnn(
                                                    self.cell, 
                                                    self.rnn_inputs_ls[i],
                                                    initial_state=init_state)


            attn_rel_ls.append([tf.reshape(tf.nn.softmax(tf.matmul(rnn_output, self.W_P) + self.b_P)[0], [-1,1])
                                            for rnn_output in rnn_outputs][:i])

            # for f_adjacent_TR_only, we still create the variables but not use some of them
            attn_TR_ls.append([tf.reshape(tf.nn.softmax(tf.matmul(rnn_output, self.W_TR) + self.b_TR)[0], [-1,1])
                                            for rnn_output in rnn_outputs][:i])

            if self.f_non_Markovian:
                cur_attn_TR_prime_ls = []
                for j in range(1, i):
                    for k in range(j):
                        if self.f_adjacent_TR_only and k != j-1:
                            # for f_adjacent_TR_only, we do not create the variables
                            continue
                        x = tf.stack([rnn_outputs[j], rnn_outputs[k]], axis=1)
                        x = tf.reshape(x, (self.rnn_batch_size, -1))
                        x = tf.nn.softmax(tf.matmul(x, self.W_TR_prime) + self.b_TR_prime)
                        cur_attn_TR_prime_ls.append(tf.reshape(x[0], [-1,1]))

                attn_TR_prime_ls.append(cur_attn_TR_prime_ls)


        return attn_rel_ls, attn_TR_ls, attn_TR_prime_ls, attn_ruleLen


    def build_inputs(self):
        rel_dict, TR_dict, alpha_dict = {}, {}, {}

        TR_ls_len = []
        for l in range(self.max_explore_len):   
            # l = rule_len - 1
            if self.f_non_Markovian:  
                cur_TR_ls_len = l+1 + int(l>0) if self.f_adjacent_TR_only else l+1 + l*(l+1)//2
            else:
                cur_TR_ls_len = l+1
            
            rel_dict[l] = tf.placeholder(tf.int64, shape=(None, l+1))
            TR_dict[l] = tf.placeholder(tf.int64, shape=(None, cur_TR_ls_len))
            alpha_dict[l] = tf.placeholder(tf.float32, shape=(None, 1))
            TR_ls_len.append(cur_TR_ls_len)

        return rel_dict, TR_dict, alpha_dict, TR_ls_len


    def calculate_TRL_score(self, rel_dict, TR_dict, alpha_dict, var_attn_rel_ls, var_attn_TR_ls, 
                                var_attn_ruleLen, var_attn_TR_prime_ls):
        score = tf.constant(0.)
        for l in range(self.max_explore_len):
            # l = rule_len - 1
            score_rel, score_TR = tf.constant(1.), tf.constant(1.)
            for k in range(l+1):
                score_rel *= tf.nn.embedding_lookup(var_attn_rel_ls[l][k], rel_dict[l][:,k])
                if self.f_non_Markovian and self.f_adjacent_TR_only and k!=0 and k!=l:
                    continue
                # TR_dict[l]: TR(I_1, I_q), TR(I_N, I_q), TR(I_1, I_2), TR(I_2, I_3), ...   if f_adjacent_TR_only
                score_TR *=  tf.nn.embedding_lookup(var_attn_TR_ls[l][k], TR_dict[l][:, min(k, 1)]) if self.f_adjacent_TR_only \
                                    else tf.nn.embedding_lookup(var_attn_TR_ls[l][k], TR_dict[l][:, k])

            if self.f_non_Markovian and l>0:
                num_TR_non_Markovian = l if self.f_adjacent_TR_only else l*(l+1)//2
                for k in range(num_TR_non_Markovian):
                    score_TR *= tf.nn.embedding_lookup(var_attn_TR_prime_ls[l][k], TR_dict[l][:, 2+k]) if self.f_adjacent_TR_only \
                                            else tf.nn.embedding_lookup(var_attn_TR_prime_ls[l][k], TR_dict[l][:, l+1+k])

            cur_score = score_rel * score_TR * alpha_dict[l]
            cur_score = tf.reduce_sum(cur_score, 0) * var_attn_ruleLen[0, l]
            score += cur_score

        return score


    def calculate_shallow_score(self):
        self.shallow_rule_idx = tf.placeholder(tf.float32, shape=(None, self.shallow_score_length))
        self.shallow_rule_alpha = tf.placeholder(tf.float32, shape=(None, self.shallow_score_length))
        self.shallow_score = tf.Variable(np.random.randn(
                                            self.num_rel,
                                            self.shallow_score_length), 
                                            dtype=tf.float32)

        attn_shallow_score = tf.nn.softmax(self.shallow_score, axis=1)
        self.rel_idx = tf.placeholder(tf.int64, shape=(None, ))
        score = tf.nn.embedding_lookup(attn_shallow_score, self.rel_idx) * self.shallow_rule_idx * self.shallow_rule_alpha
        score = tf.reduce_sum(score, 1)
        return score, attn_shallow_score


    def build_model(self, train_edges, targ_rel_ls):
        var_attn_rel_ls, var_attn_TR_ls, var_attn_TR_prime_ls, var_attn_ruleLen = self.build_rnn_graph()
        rel_dict, TR_dict, alpha_dict, TR_ls_len = self.build_inputs()

        # build the calculation graph
        query_score = self.calculate_TRL_score(rel_dict, TR_dict, alpha_dict, var_attn_rel_ls, var_attn_TR_ls, var_attn_ruleLen, var_attn_TR_prime_ls)

        valid_train_idx = self.obtain_valid_train_idx(range(len(train_edges)))

        self.shallow_rule_dict = {}
        for rel_idx in targ_rel_ls:
            batch_idx = []
            for idx in valid_train_idx:
                if train_edges[idx][1] == rel_idx:
                    batch_idx.append(idx)
            self.collect_rules(batch_idx, rel_idx)

        # add shallow layers to enhance expressiveness
        query_score_shallow, var_attn_shallow_score = self.calculate_shallow_score()
        query_score = (1-self.gamma_shallow) * query_score + (self.gamma_shallow) * query_score_shallow

        final_loss = -tf.math.log(query_score)

        optimizer = tf.train.AdamOptimizer()
        gvs = optimizer.compute_gradients(final_loss)
        optimizer_step = optimizer.apply_gradients(gvs)
        init = tf.global_variables_initializer()

        if self.f_non_Markovian:
            feed_list = [var_attn_rel_ls, var_attn_TR_ls, var_attn_TR_prime_ls, var_attn_ruleLen, var_attn_shallow_score, final_loss, optimizer_step]
        else:
            feed_list = [var_attn_rel_ls, var_attn_TR_ls, var_attn_ruleLen, var_attn_shallow_score, final_loss, optimizer_step]

        return rel_dict, TR_dict, alpha_dict, TR_ls_len, valid_train_idx, feed_list, init



    def TRL_model_training_v2(self, targ_rel_ls, num_epoch, train_edges, const_pattern_ls):
        rel_dict, TR_dict, alpha_dict, TR_ls_len, valid_train_idx, feed_list, init = self.build_model(train_edges, targ_rel_ls)
        
        loss_hist = []
        loss_prev = 100
        res_attn_rel_dict, res_attn_TR_dict, res_attn_TR_prime_dict, res_attn_ruleLen_dict = {}, {}, {}, {}
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epoch):
                loss_avg = 0.
                num_samples = 0.

                epoch_train_idx = valid_train_idx if len(valid_train_idx)>0 else list(range(len(train_edges)))
                np.random.shuffle(epoch_train_idx)
                epoch_train_idx = epoch_train_idx[: self.num_train_samples_max]

                for rel_idx in tqdm(targ_rel_ls, desc='Processing: '):
                    batch_idx = []
                    for idx in epoch_train_idx:
                        if train_edges[idx][1] == rel_idx:
                            batch_idx.append(idx)

                    batch_num = len(batch_idx)//self.batch_size
                    batch_num += 1 if len(batch_idx) % self.batch_size >0 else 0

                    for i in range(batch_num):
                        input_idx_ls = batch_idx[i*self.batch_size:(i+1)*self.batch_size]
                        cur_input_dict = {}
                        cur_input_dict[self.queries] = [[rel_idx] * (self.num_ruleLen-1) + [self.num_rel]]
                        x, f_valid, shallow_rule_idx, shallow_rule_alpha = self.prepare_inputs(input_idx_ls, const_pattern_ls, rel_idx, TR_ls_len)

                        if not f_valid:
                            continue

                        for l in range(self.max_explore_len):
                            cur_input_dict[rel_dict[l]], cur_input_dict[TR_dict[l]], cur_input_dict[alpha_dict[l]] = x[l]['rel'], x[l]['TR'], x[l]['alpha']
                            # print(x[l])

                        cur_input_dict[self.rel_idx], cur_input_dict[self.shallow_rule_idx], cur_input_dict[self.shallow_rule_alpha]  = [rel_idx], shallow_rule_idx, shallow_rule_alpha

                        if self.f_non_Markovian:
                            res_attn_rel_dict[rel_idx], res_attn_TR_dict[rel_idx], res_attn_TR_prime_dict[rel_idx], \
                                res_attn_ruleLen_dict[rel_idx], res_attn_shallow_score, loss, _ = sess.run(feed_list, feed_dict=cur_input_dict)
                        else:
                            res_attn_rel_dict[rel_idx], res_attn_TR_dict[rel_idx], \
                                    res_attn_ruleLen_dict[rel_idx], res_attn_shallow_score, loss, _ = sess.run(feed_list, feed_dict=cur_input_dict)

                        loss_avg += loss
                        num_samples += 1

                if num_samples == 0:
                    print('Error: num_samples = 0')
                    return [0]

                loss_avg = loss_avg/num_samples
                loss_avg = loss_avg[0]

                # if epoch % 20 == 0:
                #     print('Epoch ' + str(epoch+1)+ ', loss: ' + str(loss_avg))

                print('Epoch ' + str(epoch+1)+ ', loss: ' + str(loss_avg))

                if (abs(loss_prev - loss_avg) < 1e-5) and (epoch > self.num_epoch_min) and (loss_avg > loss_prev):
                    break
                loss_prev = copy.copy(loss_avg)
                loss_hist.append(loss_avg)
        
        self.save_weighs(targ_rel_ls, res_attn_rel_dict, res_attn_TR_dict, res_attn_TR_prime_dict, res_attn_ruleLen_dict, res_attn_shallow_score)

        return loss_hist



    def variable_init_tfm(self):
        var_W_rec = tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32)
        var_b_rec = tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32)

        var_W_order_ls = []
        var_b_order_ls = []
        var_W_pair_ls = []
        var_b_pair_ls = []

        for i in range(self.num_rel):
            var_W_order_ls.append(tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32))
            var_b_order_ls.append(tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32))
            var_W_pair_ls.append(tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32))
            var_b_pair_ls.append(tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32))

        gamma_tfm = tf.nn.softmax(tf.Variable(np.random.rand(1,3), dtype=tf.float32), axis=1)

        return var_W_rec, var_b_rec, var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm


    def calculate_tfm_score_v2(self, query_rel, query_rel_one_hot, related_rel_dict, h_rec, h_order, h_pair, f_exist, mask,
                               var_W_rec, var_b_rec, var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm):
        # query_rel_flatten = tf.reshape(query_rel, (-1,))
        # x1_W = tf.nn.embedding_lookup(var_W_rec, query_rel_flatten)
        # x1_W = tf.reshape(x1_W, (-1, 1))
        # x1_b = tf.nn.embedding_lookup(var_b_rec, query_rel_flatten)
        # x1_b = tf.reshape(x1_b, (-1, 1))
        # x1 =  tf.reduce_sum(x1_W* h_rec + x1_b, 1)
        x1 = tf.reduce_sum(h_rec, 1)

        x2 = tf.constant(0.)
        related_rel_dict_flatten = tf.reshape(related_rel_dict, (-1,))
        for i in range(self.num_rel):
            x2_W = tf.nn.embedding_lookup(var_W_order_ls[i], related_rel_dict_flatten)
            x2_W = tf.reshape(x2_W, (-1, self.num_rel))
            x2_W = tf.nn.softmax(x2_W, axis=1)
            # x2_b = tf.nn.embedding_lookup(var_b_order_ls[i], related_rel_dict_flatten)
            # x2_b = tf.reshape(x2_b, (-1, 1))
            # x2 += tf.reduce_sum(x2_W * (h_order + x2_b) * mask, 1) * query_rel_one_hot[:,i]
            x2 += tf.reduce_sum(x2_W * h_order * mask, 1) * query_rel_one_hot[:,i]


        x3 = tf.constant(0.)
        for i in range(self.num_rel):
            x3_W = tf.nn.embedding_lookup(var_W_pair_ls[i], related_rel_dict_flatten)
            x3_W = tf.reshape(x3_W, (-1, self.num_rel))
            x3_W = tf.nn.softmax(x3_W, axis=1)
            # x3_b = tf.nn.embedding_lookup(var_b_pair_ls[i], related_rel_dict_flatten)
            # x3_b = tf.reshape(x3_b, (-1, 1))
            # x3 += tf.reduce_sum(x3_W * (h_pair + x3_b) * mask, 1) * query_rel_one_hot[:,i]
            x3 += tf.reduce_sum(x3_W * h_pair * mask, 1) * query_rel_one_hot[:,i]

        tfm_score = tf.reduce_sum(gamma_tfm[0,0]*x1*f_exist[:,0] + \
                            gamma_tfm[0,1]*x2*f_exist[:,1] + gamma_tfm[0,2]*x3*f_exist[:,2], 0)/self.batch_size

        return tfm_score



    def build_model_tfm(self, train_edges):
        var_W_rec, var_b_rec, var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm = self.variable_init_tfm()

        query_rel = tf.placeholder(tf.int64, shape=(None, 1))
        query_rel_one_hot = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        related_rel_dict = tf.placeholder(tf.int64, shape=(None, self.num_rel))
        h_rec = tf.placeholder(tf.float32, shape=(None, 1))
        h_order = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        h_pair = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        f_exist = tf.placeholder(tf.float32, shape=(None, 3))
        mask = tf.placeholder(tf.float32, shape=(None, self.num_rel))

        valid_train_idx = self.obtain_valid_train_idx(range(len(train_edges)))
        query_score = self.calculate_tfm_score_v2(query_rel, query_rel_one_hot, related_rel_dict, h_rec, h_order, h_pair, 
                                              f_exist, mask, var_W_rec, var_b_rec,
                                              var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm)
        final_loss = -tf.math.log(query_score)

        optimizer = tf.train.AdamOptimizer(0.001)
        gvs = optimizer.compute_gradients(final_loss)
        optimizer_step = optimizer.apply_gradients(gvs)

        feed_list = [var_W_rec, var_b_rec, var_W_order_ls, var_b_order_ls, var_W_pair_ls, \
                                    var_b_pair_ls, gamma_tfm, final_loss, optimizer_step]
        

        init = tf.global_variables_initializer()
        return query_rel, query_rel_one_hot, related_rel_dict, h_rec, h_order, h_pair, f_exist, mask, valid_train_idx, feed_list, init



    def train_tfm_v2(self, rel_idx_ls, num_training, train_edges, dist_pars):
        p_rec, p_order, mu_pair, sigma_pair, lambda_pair = dist_pars
        query_rel, query_rel_one_hot, related_rel_dict, h_rec, h_order, h_pair, f_exist, mask, valid_train_idx, feed_list, init = self.build_model_tfm(train_edges)

        loss_avg_old = 100
        with tf.Session() as sess:
            sess.run(init)
            for cnt in range(num_training):
                loss_avg = 0.
                num_samples = 0.

                y = valid_train_idx if len(valid_train_idx)>0 else list(range(len(train_edges)))
                np.random.shuffle(y)
                y = y[:self.num_train_samples_max]

                batch_idx = []
                for idx in y:
                    if train_edges[idx][1] in rel_idx_ls:
                        batch_idx.append(idx)

                batch_num = len(batch_idx)//self.batch_size
                if len(batch_idx) % self.batch_size >0:
                    batch_num += 1

                for i in range(batch_num):
                    input_idx_ls = batch_idx[i*self.batch_size:(i+1)*self.batch_size]
                    cur_input_dict = {}
                    inputs = self.prepare_inputs_tfm_v2(train_edges, input_idx_ls, p_rec, p_order, mu_pair, sigma_pair, lambda_pair)
                    cur_input_dict[query_rel] = inputs['query_rel']
                    cur_input_dict[query_rel_one_hot] = inputs['query_rel_one_hot']
                    cur_input_dict[related_rel_dict] = inputs['related_rel_dict']
                    cur_input_dict[h_rec] = inputs['h_rec']
                    cur_input_dict[h_order] = inputs['h_order']
                    cur_input_dict[h_pair] = inputs['h_pair']
                    cur_input_dict[f_exist] = inputs['f_exist']
                    cur_input_dict[mask] = inputs['mask']

                    res_W_rec, res_b_rec, res_W_order_ls, res_b_order_ls, res_W_pair_ls, \
                            res_b_pair_ls, res_gamma_tfm, loss, _ = sess.run(feed_list, feed_dict=cur_input_dict)

                    loss_avg += loss
                    num_samples += 1

                if num_samples == 0:
                    print('num_samples = 0 for ' + str(rel_idx_ls))
                    return np.array([0])

                loss_avg = loss_avg/num_samples

                if cnt % 10 == 0:
                    print(str(cnt)+ ', loss:', loss_avg)

                # print(str(cnt)+ ', loss:', loss_avg)

                if (abs(loss_avg_old - loss_avg) < 1e-5) and (cnt > self.num_epoch_min) and (loss_avg > loss_avg_old):
                    break
                loss_avg_old = copy.copy(loss_avg)

        my_res = {}
        my_res['W_order_ls'] = self.my_convert_to_list(res_W_order_ls)
        my_res['W_pair_ls'] = self.my_convert_to_list(res_W_pair_ls)
        my_res['gamma_tfm'] = res_gamma_tfm.tolist()


        if self.save_weights:
            cur_path = '../output/train_weights_tfm/train_weights_'+ self.dataset_using +'_tfm.json'

            with open(cur_path, 'w') as f:
                json.dump(my_res, f)

        return loss_avg



    def variable_init_tfm_Wc(self):
        var_W_order_ls = []
        var_b_order_ls = []
        var_W_pair_ls = []
        var_b_pair_ls = []

        for i in range(self.num_rel):
            var_W_order_ls.append(tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32))
            var_b_order_ls.append(tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32))
            var_W_pair_ls.append(tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32))
            var_b_pair_ls.append(tf.Variable(np.random.rand(self.num_rel,1), dtype=tf.float32))

        gamma_tfm = tf.nn.softmax(tf.Variable(np.random.rand(1,2), dtype=tf.float32), axis=1)

        return var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm


    def calculate_tfm_Wc_score_v2(self, query_rel_one_hot, related_rel_dict, h_order, h_pair, f_exist, mask,
                                  var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm):
        x2 = tf.constant(0.)
        related_rel_dict_flatten = tf.reshape(related_rel_dict, (-1,))
        for i in range(self.num_rel):
            x2_W = tf.nn.embedding_lookup(var_W_order_ls[i], related_rel_dict_flatten)
            x2_W = tf.reshape(x2_W, (-1, self.num_rel))
            x2_W = tf.nn.softmax(x2_W, axis=1)
            # x2_b = tf.nn.embedding_lookup(var_b_order_ls[i], related_rel_dict_flatten)
            # x2_b = tf.reshape(x2_b, (-1, 1))
            # x2 += tf.reduce_sum(x2_W * (h_order + x2_b) * mask, 1) * query_rel_one_hot[:,i]
            x2 += tf.reduce_sum(x2_W * h_order * mask, 1) * query_rel_one_hot[:,i]

        x3 = tf.constant(0.)
        for i in range(self.num_rel):
            x3_W = tf.nn.embedding_lookup(var_W_pair_ls[i], related_rel_dict_flatten)
            x3_W = tf.reshape(x3_W, (-1, self.num_rel))
            x3_W = tf.nn.softmax(x3_W, axis=1)
            # x3_b = tf.nn.embedding_lookup(var_b_pair_ls[i], related_rel_dict_flatten)
            # x3_b = tf.reshape(x3_b, (-1, 1))
            # x3 += tf.reduce_sum(x3_W * (h_pair + x3_b) * mask, 1) * query_rel_one_hot[:,i]
            x3 += tf.reduce_sum(x3_W * h_pair * mask, 1) * query_rel_one_hot[:,i]

        tfm_score = tf.reduce_sum(gamma_tfm[0,0]*x2*f_exist[:,0] + gamma_tfm[0,1]*x3*f_exist[:,1], 0)/self.batch_size

        return tfm_score



    def build_model_tfm_Wc(self, train_edges):
        var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm = self.variable_init_tfm_Wc()

        query_rel_one_hot = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        related_rel_dict = tf.placeholder(tf.int64, shape=(None, self.num_rel))
        h_order = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        h_pair = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        f_exist = tf.placeholder(tf.float32, shape=(None, 2))
        mask = tf.placeholder(tf.float32, shape=(None, self.num_rel))


        valid_train_idx = self.obtain_valid_train_idx(range(len(train_edges)))
        query_score = self.calculate_tfm_Wc_score_v2(query_rel_one_hot, related_rel_dict, h_order, h_pair, f_exist, mask,
                                                     var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm)
        final_loss = -tf.math.log(query_score)

        optimizer = tf.train.AdamOptimizer(0.001)
        gvs = optimizer.compute_gradients(final_loss)
        optimizer_step = optimizer.apply_gradients(gvs)

        feed_list = [var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm, final_loss, optimizer_step]

        init = tf.global_variables_initializer()
        return query_rel_one_hot, related_rel_dict, h_order, h_pair, f_exist, mask, valid_train_idx, feed_list, init



    def train_tfm_Wc_v2(self, rel_idx_ls, num_training, train_edges, dist_pars):
        query_rel_one_hot, related_rel_dict, h_order, h_pair, f_exist, mask, valid_train_idx, feed_list, init = self.build_model_tfm_Wc(train_edges)
        p_order, mu_pair, sigma_pair, lambda_pair = dist_pars

        loss_avg_old = 100
        with tf.Session() as sess:
            sess.run(init)
            for cnt in range(num_training):
                loss_avg = 0.
                num_samples = 0.

                y = list(range(len(train_edges)))
                if len(valid_train_idx)>0:
                    y = valid_train_idx

                np.random.shuffle(y)
                y = y[:self.num_train_samples_max]

                batch_idx = []
                for idx in y:
                    if train_edges[idx][1] in rel_idx_ls:
                        batch_idx.append(idx)

                batch_num = len(batch_idx)//self.batch_size
                if len(batch_idx) % self.batch_size >0:
                    batch_num += 1

                for i in range(batch_num):
                    input_idx_ls = batch_idx[i*self.batch_size:(i+1)*self.batch_size]
                    cur_input_dict = {}
                    x = self.prepare_inputs_tfm_Wc_v2(train_edges, input_idx_ls, p_order, mu_pair, sigma_pair, lambda_pair)
                    cur_input_dict[query_rel_one_hot] = x['query_rel_one_hot']
                    cur_input_dict[related_rel_dict] = x['related_rel_dict']
                    cur_input_dict[h_order] = x['h_order']
                    cur_input_dict[h_pair] = x['h_pair']
                    cur_input_dict[f_exist] = x['f_exist']
                    cur_input_dict[mask] = x['mask']

                    res_W_order_ls, res_b_order_ls, res_W_pair_ls, \
                            res_b_pair_ls, res_gamma_tfm, loss, _ = sess.run(feed_list, feed_dict=cur_input_dict)

                    loss_avg += loss
                    num_samples += 1

                if num_samples == 0:
                    print('num_samples = 0', rel_idx_ls)
                    return np.array([0])

                loss_avg = loss_avg/num_samples

                if cnt % 10 == 0:
                    print(str(cnt)+ ', loss:', loss_avg)

                if (abs(loss_avg_old - loss_avg) < 1e-5) and (cnt > self.num_epoch_min) and (loss_avg > loss_avg_old):
                    break
                loss_avg_old = copy.copy(loss_avg)

        my_res = {}
        my_res['W_order_ls'] = self.my_convert_to_list(res_W_order_ls)
        my_res['W_pair_ls'] = self.my_convert_to_list(res_W_pair_ls)
        my_res['gamma_tfm_Wc'] = res_gamma_tfm.tolist()


        if self.save_weights:
            cur_path = '../output/train_weights_tfm/train_weights_'+ self.dataset_using +'_tfm_Wc.json'

            with open(cur_path, 'w') as f:
                json.dump(my_res, f)

        return loss_avg


    # def calculate_query_score(self, edges, ent_walk_res, query_int, var_prob_rel_ls, var_prob_pattern_ls):
    #     path_len = ent_walk_res.shape[1]-1
    #     prob_path_ls = []
    #     for l in range(ent_walk_res.shape[0]):
    #         cur_prob_path = tf.constant(1.)
    #         for k in range(path_len):
    #             rel, tmp_rel = self.obtain_edges_given_nodes(edges, ent_walk_res[l,k], ent_walk_res[l,k+1], query_int)

    #             prob_rel = tf.nn.embedding_lookup(var_prob_rel_ls[path_len-1][k], rel)
    #             prob_pattern = tf.nn.embedding_lookup(var_prob_pattern_ls[path_len-1][k], tmp_rel)
    #             cur_prob_path = cur_prob_path * tf.reduce_sum(tf.nn.softmax(prob_rel * prob_pattern) * prob_rel * prob_pattern, 0)

    #         prob_path_ls.append(cur_prob_path)

    #     prob_path = tf.concat(prob_path_ls, 0)
    #     prob_query = tf.reduce_sum(tf.nn.softmax(prob_path) * prob_path, 0)

    #     return prob_query


    # def calculate_query_score_fun_ver2(self, rel_dict, tmp_rel_dict, rPro_dict, var_prob_rel_ls, var_prob_pattern_ls, var_prob_ruleLen):
    #     prob = tf.constant(0.)
    #     for l in range(self.max_explore_len):
    #         prob_rel = tf.constant(1.)
    #         # print(l)
    #         for k in range(l+1):
    #             prob_rel = prob_rel * tf.nn.embedding_lookup(var_prob_rel_ls[l][k], rel_dict[l][:,k])
    #         # print(prob_rel.shape)

    #         prob_pattern = tf.constant(1.)
    #         for k in range((l+1)*(l+2)//2):
    #             prob_pattern = prob_pattern * tf.nn.embedding_lookup(var_prob_pattern_ls[l][k], tmp_rel_dict[l][:,k])
    #         # print(prob_pattern.shape)

    #         cur_prob = prob_rel * prob_pattern * rPro_dict[l]
    #         # print(cur_prob.shape)

    #         cur_prob = tf.reduce_sum(cur_prob, 0) * var_prob_ruleLen[0, l]
    #         # print(cur_prob.shape)

    #         prob += cur_prob

    #     return prob


    # def variable_init_v3(self):
    #     prob_rel_ls = []
    #     prob_pattern_ls = []
    #     prob_ruleLen = tf.nn.softmax(tf.Variable(np.random.rand(1, self.num_ruleLen), dtype=tf.float32), axis=1)

    #     for i in range(1, self.num_ruleLen+1):
    #         prob_rel = tf.nn.softmax(tf.Variable(np.random.rand(self.num_rel, i), dtype=tf.float32), axis=0)
    #         prob_rel = tf.split(prob_rel, i, axis=1)
    #         prob_rel_ls.append(prob_rel)

    #         prob_pattern = tf.nn.softmax(tf.Variable(np.random.rand(self.num_pattern, i), dtype=tf.float32), axis=0)
    #         prob_pattern = tf.split(prob_pattern, i, axis=1)
    #         prob_pattern_ls.append(prob_pattern)

    #     return prob_rel_ls, prob_pattern_ls, prob_ruleLen








class Collector(gadgets):
    def __init__(self, num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode):
        super(Collector, self).__init__(num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode)
    
    def create_all_rule_dicts(self, rel_idx, train_edges, query_idx=None, pos_examples_idx=None):
        assert self.overall_mode in ['general', 'few', 'biased', 'time_shifting']
        
        rule_dict, rule_sup_num_dict = {}, {}
        cnt = 0

        idx_ls = range(len(train_edges))
        query_idx = range(len(train_edges)) if query_idx is None else query_idx

        for idx1 in query_idx:
            idx = idx_ls[idx1]
            if isinstance(pos_examples_idx, list) and (idx not in pos_examples_idx):
                continue
            if train_edges[idx][1] != rel_idx:
                continue
            
            path = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(idx)+'.json' if self.overall_mode == 'general' else\
                   '../output/found_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'

            if not os.path.exists(path):
                # print(path + ' not found')
                continue

            with open(path, 'r') as f:
                data = json.load(f)

            cnt += 1
            for k in data.keys():
                if int(k) > self.max_explore_len:
                    continue

                cur_rule_mat = np.vstack([r['rule'] for r in data[k]]) if len(data[k]) >1 else np.array([r['rule'] for r in data[k]])
                cur_rule_mat = np.unique(cur_rule_mat, axis=0)

                rule_dict[k] = cur_rule_mat.copy() if k not in rule_dict.keys() else np.vstack((rule_dict[k], cur_rule_mat))
                rule_dict[k] = np.unique(rule_dict[k], axis=0)

                rule_sup_num_dict = self.my_merge_dict(rule_sup_num_dict, dict(Counter([str(r['rule']) for r in data[k]])))

        return rule_dict, rule_sup_num_dict

    def read_weight(self, rel_idx):
        # calculate rule weights
        cur_path = self.get_weights_savepath_v2(rel_idx)

        if not os.path.exists(cur_path):
            # print(cur_path + ' not found')
            return None, None, None, None, None
        
        with open(cur_path, 'r') as f:
            my_res = json.load(f)

        var_prob_rel_ls = [np.squeeze(np.array(x), 2) for x in my_res['attn_rel_ls']]
        var_prob_pattern_ls = [np.squeeze(np.array(x), 2) for x in my_res['attn_TR_ls']]
        var_prob_pattern_prime_ls = [[]] + [np.squeeze(np.array(x), 2) for x in my_res['attn_TR_prime_ls'][1:]] if self.f_non_Markovian else []
        var_prob_ruleLen = np.array(my_res['attn_ruleLen'])

        return var_prob_rel_ls, var_prob_pattern_ls, var_prob_pattern_prime_ls, var_prob_ruleLen, my_res


    def calculate_rule_score(self, rule, ruleLen, const_pattern_ls, var_prob_rel_ls, var_prob_pattern_ls, var_prob_pattern_prime_ls, var_prob_ruleLen):
        rel_dict = rule[:ruleLen]
        tmp_rel_dict = rule[ruleLen:]
        tmp_rel_dict = [const_pattern_ls.index(TR) for TR in tmp_rel_dict]
        prob_rel = 1.
        prob_pattern = 1.
        for k in range(ruleLen):
            prob_rel *= var_prob_rel_ls[ruleLen-1][k, rel_dict[k]]
            if self.f_non_Markovian and self.f_adjacent_TR_only and k!=0 and k!= ruleLen-1:
                continue
            prob_pattern *= var_prob_pattern_ls[ruleLen-1][k, tmp_rel_dict[min(k, 1)]] if self.f_adjacent_TR_only else var_prob_pattern_ls[ruleLen-1][k, tmp_rel_dict[k]]

        if self.f_non_Markovian and ruleLen > 1:
            num_TR_non_Markovian = ruleLen-1 if self.f_adjacent_TR_only else ruleLen*(ruleLen - 1)//2
            for k in range(num_TR_non_Markovian):
                prob_pattern *= var_prob_pattern_prime_ls[ruleLen-1][k, tmp_rel_dict[2+k]] if self.f_adjacent_TR_only \
                                        else var_prob_pattern_prime_ls[ruleLen-1][k, tmp_rel_dict[ruleLen+k]]

        prob = prob_rel * prob_pattern * var_prob_ruleLen[0, ruleLen-1]

        return prob


    def calculate_rule_score_from_res(self, rule, rule_Len, const_pattern_ls, var_prob_rel_ls, var_prob_pattern_ls, var_prob_pattern_prime_ls, var_prob_ruleLen, my_res):
        s = self.calculate_rule_score(rule, int(rule_Len), const_pattern_ls, var_prob_rel_ls, var_prob_pattern_ls, 
                                                    var_prob_pattern_prime_ls, var_prob_ruleLen)
        s *= (1-self.gamma_shallow)
        s += (self.gamma_shallow) * my_res['shallow_score'][my_res['shallow_rule_dict'].index(rule)] if rule in my_res['shallow_rule_dict'] else 0
        return s


    def write_all_rule_dicts(self, rel_idx, rule_dict, rule_sup_num_dict, const_pattern_ls, cal_score=False, select_topK_rules=True):
        if cal_score:
            var_prob_rel_ls, var_prob_pattern_ls, var_prob_pattern_prime_ls, var_prob_ruleLen, my_res = self.read_weight(rel_idx)
            if my_res is None:
                return

        rule_dict1 = {}
        for rule_Len in rule_dict.keys():
            rule_dict1[rule_Len] = []
            for r in rule_dict[rule_Len]:
                r = r.tolist()
                cur_rule_dict = {'rule': r, 'sup_num': rule_sup_num_dict[str(r)]}
                if cal_score:
                    s = self.calculate_rule_score_from_res(r, rule_Len, const_pattern_ls, var_prob_rel_ls, var_prob_pattern_ls, var_prob_pattern_prime_ls, var_prob_ruleLen, my_res)
                    cur_rule_dict['score'] = s

                rule_dict1[rule_Len].append(cur_rule_dict)
                
            if select_topK_rules:
                rule_dict1[rule_Len] = sorted(rule_dict1[rule_Len], key=lambda x: x['sup_num'], reverse=True)[:self.max_rulenum[int(rule_Len)]]

        path = '../output/learned_rules/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json' if self.overall_mode == 'general' else\
               '../output/learned_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json'
        
        with open(path, 'w') as f:
            json.dump(rule_dict1, f)
        
        return 


    # def make_specific_rules(self, rule_walks, mode):
    #     dur_list = [(0,6), (6,11), (11,21), (21,10000)]
    #     if 'dur' in mode:
    #         mode_t = []
    #         for i in range(rule_walks.shape[1]//2):
    #             dur = rule_walks[:,2*i+1] - rule_walks[:,2*i]
    #             res = dur.copy()
    #             # print(res)
    #             for j in range(len(dur_list)):
    #                 res[(dur>=dur_list[j][0]) & (dur<dur_list[j][1])] = j
    #             # print(res)
    #             mode_t.append(res.reshape(-1,1))
    #         if len(mode_t) == 1:
    #             mode_t = mode_t[0].tolist()
    #         else:
    #             mode_t = np.hstack(mode_t).tolist()

    #         # print(mode_t)
    #         c = Counter([str(t) for t in mode_t])
    #         c = dict(c)
    #         s = sum(c.values())
    #         for k in c.keys():
    #             c[k] = c[k]/s
    #         # print(c)
    #     return c



    # def create_rule_supplement(self, rel_idx, train_edges, query_idx=None):
    #     with open('../output/learned_rules/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json','r') as f:
    #         rule_dict1 = json.load(f)

    #     with open('../output/'+ self.dataset_using +'_contri_dict_'+str(rel_idx)+'.json','r') as f:
    #         contri_dict = json.load(f)

    #     if not query_idx:
    #         query_idx = range(len(train_edges))

    #     rule_sup = {}
    #     for rule_Len in rule_dict1.keys():
    #         if int(rule_Len)>1:
    #             for r in rule_dict1[rule_Len]:
    #                 res_dict = {}
    #                 for idx in query_idx:
    #                     if train_edges[idx][1] == rel_idx:
    #                         line = train_edges[idx]
    #                         with open('../output/found_rules/'+ self.dataset_using \
    #                                             +'_train_query_'+str(idx)+'.json', 'r') as f:
    #                             v = json.load(f)
    #                         if rule_Len in v.keys():
    #                             cur_rules = [w['rule'] for w in v[rule_Len]]
    #                             # print(cur_rules)
    #                             if r['rule'] in cur_rules:
    #                                 # print(line)
    #                                 masked_facts = np.delete(train_edges, \
    #                                             [idx, self.get_inv_idx(len(train_edges)//2, idx)], 0)
    #                                 X = self.create_mapping_facts(masked_facts, line[3:], 1)
    #                                 x = np.unique(X[:,:4], axis=0)
    #                                 _, rule_walks = self.apply_single_rule(line[0], line[3:], r['rule'], int(rule_Len),
    #                                                                                     x, return_walk=1)
    #                                 rule_walks = self.obtain_sp_rule_walks(rule_walks, r['rule'], int(rule_Len), X)
    #                                 # print(rule_walks)
    #                                 y = rule_walks[rule_walks[:,-1]==line[2],:-1].tolist()
    #                                 cur_dict = dict(Counter([tuple(y1) for y1 in y]))
    #                                 res_dict = self.my_merge_dict(res_dict, cur_dict)

    #                     rule_sup[str(r['rule'])] = res_dict
    #     return rule_sup





class Predictor(Walker):
    def __init__(self, num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode):
        super(Predictor, self).__init__(num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode)

    def create_rule_score_bar(self, rule_dict1):
        rule_score_bar = {}
        for rule_Len in rule_dict1.keys():
            r_score_ls = []
            for r in rule_dict1[rule_Len]:
                r_score_ls.append(r['score'])
            r_score_ls.sort(reverse=True)
            rule_score_bar[rule_Len] = r_score_ls[min(len(r_score_ls)-1, self.max_rulenum[int(rule_Len)])]

        return rule_score_bar


    def create_res_ts_dict(self, query, rule, rule_Len, cur_walk, res_ts_dict):
        for w in cur_walk:
            if w[-1] not in res_ts_dict:
                res_ts_dict[w[-1]] = {}
            for l in range(int(rule_Len)):
                if rule['rule'][l] not in res_ts_dict[w[-1]]:
                    res_ts_dict[w[-1]][rule['rule'][l]] = []
                res_ts_dict[w[-1]][rule['rule'][l]].append(w[3*l+1]-query[3])
        return res_ts_dict


    def create_mapping_facts(self, facts, query_int, mode=0):
        x = self.obtain_tmp_rel(facts[:, 3:], query_int).reshape((-1,1))
        if mode:
            x = np.hstack((facts[:, :3], x, facts[:, 3:]))
        else:
            x = np.hstack((facts[:, :3], x))
            x = np.unique(x, axis=0)
            x = x.astype(int)
        return x


    def evaluate_res_dict(self, res_dict, targ_node, query, facts, res_tfm_score, tol_gap=[0,0]):
        res_mat = np.zeros((self.num_entites,1))
        for k in res_dict.keys():
            res_mat[k,0] = res_dict[k]

        res_mat += res_tfm_score

        s = copy.copy(res_mat[targ_node,0])

        # delete other correct answers in rank
        y = facts[np.all(facts[:,:2]==[query[0], query[1]], axis=1),3:]
        z = facts[np.all(facts[:,:2]==[query[0], query[1]], axis=1),2:3]
        y = self.obtain_tmp_rel(y, query[3:], tol_gap)
        res_mat[z[y==0]] -= 9999

        rank = len(res_mat[res_mat[:,0]>s])+1

        return rank


    def fact_check(self, f_check_enable, assiting_data, res_mat, query, facts, format_extra_len=0):
        assert self.dataset_using in ['wiki', 'YAGO']
        
        if self.dataset_using == 'wiki':
            ent_int_mat, ent_int_valid_mat, Gauss_int_dict = assiting_data
        else:
            ent_int_mat, ent_int_valid_mat, Gauss_int_dict, query_prop_dict, ent_prop_mat = assiting_data

        if f_check_enable > 0:
            if f_check_enable == 1:
                res_mat_exp = res_mat - 1
                if self.dataset_using == 'wiki':
                    res_mat[(ent_int_mat[:, 0] - 15*(10**format_extra_len) > query[3]) & (ent_int_valid_mat[:, 0] == 1)] -= 1
                    if not query[1] in [17, 20]:
                        res_mat[(ent_int_mat[:, 1] + 10*(10**format_extra_len) < query[4]) & (ent_int_valid_mat[:, 1] == 1)] -= 1                        
                else:                        
                    if not (query[3]<1000*(10**format_extra_len) and query[4]>1000*(10**format_extra_len)):
                        res_mat[(ent_int_mat[:, 0] - 15*(10**format_extra_len) > query[3]) & (ent_int_valid_mat[:, 0] == 1)] -= 1
                    res_mat[(ent_int_mat[:, 1] + 10*(10**format_extra_len) < query[4]) & (ent_int_valid_mat[:, 1] == 1)] -= 1
                res_mat = np.max(np.hstack((res_mat_exp, res_mat)), axis=1).reshape((-1,1))

            if self.dataset_using == 'wiki':
                if query[1] in Gauss_int_dict.keys():
                    x = query[3] - ent_int_mat[:, 0]
                    y = norm(Gauss_int_dict[query[1]][0], Gauss_int_dict[query[1]][1]).pdf(x).reshape((-1,1))
                    res_mat[ent_int_valid_mat[:, 0] == 1] += 0.1*y[ent_int_valid_mat[:, 0] == 1]
            else:
                rel_once = [10, 17]
                if query[1] in rel_once:
                    res_mat[facts[facts[:,1] == query[1]][:,2]] -= 1

                if query[1] in query_prop_dict['p2p'] + query_prop_dict['n2p']:
                    res_mat[ent_prop_mat[:,0] == -1] -= 1
                elif query[1] in query_prop_dict['p2n'] + query_prop_dict['u2n']:
                    res_mat[ent_prop_mat[:,0] == 1] -= 1

                if query[1] in query_prop_dict['p2p'] + query_prop_dict['n2p']:
                    y = facts[facts[:, 1]==query[1], 3:]
                    z = facts[facts[:, 1]==query[1], 2:3]
                    y = self.obtain_tmp_rel(y, query[3:]).reshape((-1,1))
                    res_mat[z[y==0]] -= 0.1

                if f_check_enable == 1:
                    if query[1] in Gauss_int_dict.keys():
                        if query[1] != 17:
                            x = query[3] - ent_int_mat[:, 0]
                            y = norm(Gauss_int_dict[query[1]][0], Gauss_int_dict[query[1]][1]).pdf(x).reshape((-1,1))
                        else:
                            x = query[3] - ent_int_mat[:, 1]
                            y = norm(Gauss_int_dict[10][0], Gauss_int_dict[10][1]).pdf(x).reshape((-1,1))
                        res_mat[ent_int_valid_mat[:, 0] == 1] += 0.1*y[ent_int_valid_mat[:, 0] == 1]

        return res_mat


    def predict_tfm_score(self, train_edges, query_rel, query_time, candidate, p_rec, p_order, mu_pair, sigma_pair, lambda_pair):
        with open('../output/train_weights_tfm/train_weights_' + self.dataset_using + '_tfm.json', 'r') as f:
            my_res = json.load(f)

        W_order_ls = np.array(my_res['W_order_ls'])
        W_pair_ls = np.array(my_res['W_pair_ls'])
        gamma_tfm = my_res['gamma_tfm']

        related_rel_dict, h_rec, h_order, h_pair = self.prepare_inputs_tfm_test(train_edges, query_rel, query_time, 
                                                            candidate, p_rec, p_order, mu_pair, sigma_pair, lambda_pair)

        x1 = h_rec
        x2 = 0
        x3 = 0
        if len(related_rel_dict)>0:
            x2 = np.sum(np.exp(W_order_ls[query_rel, related_rel_dict][0])*h_order)/np.sum(np.exp(W_order_ls[query_rel, related_rel_dict][0]))
            x3 = np.sum(np.exp(W_pair_ls[query_rel, related_rel_dict][0])*h_pair)/np.sum(np.exp(W_pair_ls[query_rel, related_rel_dict][0]))

        return x1*gamma_tfm[0][0] + x2*gamma_tfm[0][1] + x3*gamma_tfm[0][2]


    def predict_tfm_Wc_score(self, query_rel, ts_dict, p_order, mu_pair, sigma_pair, lambda_pair):
        with open('../output/train_weights_tfm/train_weights_' + self.dataset_using + '_tfm_Wc.json', 'r') as f:
            my_res = json.load(f)

        W_order_ls = np.array(my_res['W_order_ls'])
        W_pair_ls = np.array(my_res['W_pair_ls'])
        gamma_tfm = my_res['gamma_tfm_Wc']

        related_rel_dict, h_order, h_pair = self.prepare_inputs_tfm_Wc_test(query_rel, ts_dict,
                                                                         p_order, mu_pair, sigma_pair, lambda_pair)

        x2 = 0
        x3 = 0
        if len(related_rel_dict)>0:
            x2 = np.sum(np.exp(W_order_ls[query_rel, related_rel_dict][0])*h_order)/np.sum(np.exp(W_order_ls[query_rel, related_rel_dict][0]))
            x3 = np.sum(np.exp(W_pair_ls[query_rel, related_rel_dict][0])*h_pair)/np.sum(np.exp(W_pair_ls[query_rel, related_rel_dict][0]))

        return x2*gamma_tfm[0][0] + x3*gamma_tfm[0][1]


    def predict_tfm_score_with_fact_check(self, query, bg_pred, facts, res_dict, dist_pars, assiting_data, format_extra_len=0):
        f_check_enable = [0, 0.5, 1][0] # whether enable
        res_mat = np.zeros((self.num_entites,1))

        if len(dist_pars) > 0:
            if self.f_Wc_ts:
                p_rec, p_order, mu_pair, sigma_pair, lambda_pair, \
                                    p_order_Wc, mu_pair_Wc, sigma_pair_Wc, lambda_pair_Wc = dist_pars
            else:
                p_rec, p_order, mu_pair, sigma_pair, lambda_pair = dist_pars

            if query[1]>= self.num_rel//2:
                cur_query_rel = query[1] - self.num_rel//2
            else:
                cur_query_rel = query[1] + self.num_rel//2

            TRL_total_score = sum(res_dict.values())
            for cand in res_dict:
                res_mat[cand, 0] += 0.1* TRL_total_score *self.predict_tfm_score(bg_pred, 
                                                            cur_query_rel, query[3], cand, 
                                                            p_rec, p_order, mu_pair, sigma_pair, lambda_pair)
                if self.f_Wc_ts:
                    res_mat[cand, 0] += 0.05* TRL_total_score *self.predict_tfm_Wc_score(query[1], res_dict[cand], 
                                                        p_order_Wc, mu_pair_Wc, sigma_pair_Wc, lambda_pair_Wc)
        
        res_mat = self.fact_check(f_check_enable, assiting_data, res_mat, query, facts, format_extra_len=format_extra_len)
        return res_mat



    def predict(self, rel_idx, bg_pred, test_data, test_data_inv, const_pattern_ls, assiting_data, dist_pars, train_edges,
                    queries_idx = None, tol_gap=[0, 0], selected_rules=None, enable_pure_guessing=True,
                    format_extra_len=0, f_predicting=0):

        f_name = 'learned_rules' if self.overall_mode == 'general' else 'learned_rules_' + self.overall_mode
        path = '../output/'+f_name+'/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json'

        if not os.path.exists(path):
            # print(path + ' not found')
            rule_dict1 = {}
            if not enable_pure_guessing:
                return {}
        else:
            with open(path, 'r') as f:
                rule_dict1 = json.load(f)

        data_using = test_data if rel_idx < self.num_rel//2 else test_data_inv
        s = 0 if rel_idx < self.num_rel//2  else len(test_data)
        rule_score_bar = self.create_rule_score_bar(rule_dict1)
        mode = 1 if self.f_Wc_ts or self.f_non_Markovian else 0
        queries_idx = range(len(data_using)) if queries_idx is None else queries_idx

        rank_dict = {}
        for idx in queries_idx:
            if data_using[idx][1] != rel_idx:
                continue
            
            query = data_using[idx]
            cur_bg_pred = bg_pred[bg_pred[:, 3]<query[3]] if f_predicting else bg_pred
    
            res_dict, res_ts_dict = {}, {}
            x = self.create_mapping_facts(cur_bg_pred, query[3:], mode)
            for rule_Len in rule_dict1.keys():
                for r in rule_dict1[rule_Len]:
                    if isinstance(selected_rules, dict) and (not r['rule'] in selected_rules[rel_idx]):
                        continue
                    if r['score'] < rule_score_bar[rule_Len]:
                        continue

                    cur_dict, cur_walk = self.apply_single_rule(query[0], r['rule'], int(rule_Len), x, f_print=0, return_walk=True)

                    if self.f_Wc_ts:
                        res_ts_dict = self.create_res_ts_dict(query, r, rule_Len, cur_walk, res_ts_dict)

                    for k in cur_dict.keys():
                        cur_dict[k] = cur_dict[k] * r['score']

                    res_dict = self.my_merge_dict(res_dict, cur_dict)

            if enable_pure_guessing and (len(res_dict) == 0):
                for i in range(self.num_entites):
                    res_dict[i] = np.random.normal(loc=0.0, scale=0.01, size=None)

            res_tfm_score = self.predict_tfm_score_with_fact_check(query, bg_pred, train_edges, res_dict, dist_pars, assiting_data, format_extra_len)

            rank = self.evaluate_res_dict(res_dict, query[2], query, train_edges, res_tfm_score, tol_gap)
            rank_dict[idx+s] = rank

        return rank_dict


    def predict_in_batch(self, i, num_queries, num_processes, rel_idx, bg_pred, test_data, test_data_inv, 
                            const_pattern_ls, assiting_data, dist_pars, train_edges, rules_dict=None, 
                            rule_scores=None, tol_gap=[0, 0], selected_rules=None,
                            format_extra_len=0, f_predicting=0, enable_pure_guessing = True):
        n_t = len(test_data)
        num_rest_queries = n_t - (i + 1) * num_queries
        if (num_rest_queries >= num_queries) and (i + 1 < num_processes):
            queries_idx = range(i * num_queries, (i + 1) * num_queries)
        else:
            queries_idx = range(i * num_queries, n_t)

        rank_dict = self.predict(rel_idx, bg_pred, test_data, test_data_inv, const_pattern_ls, 
                                    assiting_data, dist_pars, train_edges,
                                    queries_idx, tol_gap, selected_rules, enable_pure_guessing,
                                    format_extra_len, f_predicting)

        return rank_dict