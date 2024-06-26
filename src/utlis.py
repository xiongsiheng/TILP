import random
import numpy as np
import sys
import json
import os
import pandas as pd
import copy
from collections import Counter
from scipy.stats import norm
from collections import defaultdict



class TILP(object):
    def __init__(self, num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode):
        self.num_rel = num_rel
        self.num_pattern = num_pattern
        self.num_ruleLen = num_ruleLen
        self.dataset_using = dataset_using

        if self.dataset_using == 'wiki':
            self.num_entites = 12554
            self.prob_cal_alpha = 1  # if less than 1, do sampling when calculating arriving rate for training
        elif self.dataset_using == 'YAGO':
            self.num_entites = 10623
            self.prob_cal_alpha = 1

        self.num_train_samples_max = 100000  # max number of training samples
        self.batch_size = 128
        self.num_epoch_min = 10
        self.save_weights = True

        self.max_explore_len = num_ruleLen
        self.overall_mode = overall_mode
        self.num_paths_dict = num_paths_dict

        self.rnn_query_embed_size = 128
        self.rnn_state_size = 128
        self.rnn_batch_size = 1
        self.rnn_num_layer = 1

        '''
        Different TR setting:
        f_Markovian: TR(I_1, I_q), TR(I_2, I_q), ..., TR(I_N, I_q)
        f_non_Markovian: TR(I_1, I_q), TR(I_2, I_q), ..., TR(I_N, I_q), TR(I_1, I_2), TR(I_1, I_3), TR(I_2, I_3), TR(I_1, I_4), ..., TR(I_{N-1}, I_N)
        f_non_Markovian and f_adjacent_TR_only: TR(I_1, I_q), TR(I_N, I_q), TR(I_1, I_2), TR(I_2, I_3), ..., TR(I_{N-1}, I_N)
        '''
        self.f_non_Markovian = True # consider non-Markovian constraints
        self.f_adjacent_TR_only = False # consider adjacent TRs only

        self.f_Wc_ts = False # consider intermediate nodes for temporal feature modeling
        self.max_rulenum = {1: 20, 2: 50, 3: 100, 4: 100, 5: 200} # max number of rules for each rule length

        # add shallow layers to enhance expressiveness
        self.gamma_shallow = 0.2 # weight for shallow score
        self.shallow_score_length = 400 # max number of shallow rules

        if self.overall_mode == 'general':
            self.weights_savepath = '../output/train_weights/train_weights_' + self.dataset_using
        elif self.overall_mode in ['few', 'biased', 'time_shifting']:
            self.weights_savepath = '../output/train_weights_'+ self.overall_mode +'/train_weights_'\
                                     + self.dataset_using



class gadgets(TILP):
    def count_lists(self, input_list):
        # Convert each sublist to a tuple so they can be counted
        tuples_list = [tuple(sublist) for sublist in input_list]

        # Count the occurrences of each tuple
        tuples_count = Counter(tuples_list)

        # Filter tuples that occur more than once and convert them back into lists, pairing with their counts
        output = [(item, count) for item, count in tuples_count.items()]

        sorted_list = sorted(output, key=lambda x: x[1], reverse=True)
        # Extracting the first element of each tuple for the output
        output = [item[0] for item in sorted_list]

        return output


    def obtain_tmp_rel(self, int1, int2, tol_gap=[0, 0]):
        '''
        Obtain TR between two intervals (tolerance gap is considered).
        '''
        if len(int2.shape) == 1:
            return ((int1[:, 1] <= int2[0] - tol_gap[0]) & (int1[:, 0] <= int2[0] - tol_gap[0]))*(-1) + ((int1[:, 0] >= int2[1] + tol_gap[1]) & (int1[:, 1] >= int2[1] + tol_gap[1]))*1
        else:
            return ((int1[:, 1] <= int2[:,0] - tol_gap[0]) & (int1[:, 0] <= int2[:,0] - tol_gap[0]))*(-1) + ((int1[:, 0] >= int2[:,1] + tol_gap[1]) & (int1[:, 1] >= int2[:,1] + tol_gap[1]))*1


    def calculate_intersection_length(self, interval1, interval2):
        start = np.maximum(interval1[0], interval2[0])
        end = np.minimum(interval1[1], interval2[1])
        return np.maximum(0, end - start)


    def point_in_intervals(self, intervals, point):
        results = (intervals[:, 0] <= point) & (point <= intervals[:, 1])
        return results.astype(int)


    def calculate_overlap_degree(self, known_interval, intervals):
        '''
        Given a known interval, calculate the overlap degree with other intervals.
        '''
        known_interval_length = known_interval[1] - known_interval[0]
        if known_interval_length > 0:
            intersection_lengths = self.calculate_intersection_length(known_interval, intervals.T)
            overlap_degrees = (intersection_lengths*1.) / known_interval_length
            return overlap_degrees
        else:
            return self.point_in_intervals(intervals, known_interval[0])*1.


    def print_dict(self, dict1):
        print('start')
        for k in dict1.keys():
            print(k, dict1[k])
        print('end')


    def tmp_stat(self, rel_idx, facts):
        x = facts[facts[:,1]==rel_idx]
        y = x[:,4] - x[:,3]
        # print(x[x[:,4] - x[:,3]<0])
        y = sorted(y)
        print(Counter(y))
        print(y[len(y)//2])


    def select_events(self, related_facts, query_int):
        z = np.unique(related_facts[:,:3], axis=0)
        y = []
        for edge in z:
            x = np.all(related_facts[:,:3] == edge, axis=1)
            x = related_facts[x]
            x1 = x[x[:,3] <= query_int[0]]
            f_x = 0
            if len(x1) == 0:
                x1 = x.copy()
                f_x = 1
            if f_x == 0:
                t_s = max(x1[:,3])
                t_e = max(x1[x1[:,3] == t_s][:,4])
            else:
                t_s = min(x1[:,3])
                t_e = min(x1[x1[:,3] == t_s][:,4])

            y.append([edge[0], edge[2], t_s, t_e])

        related_facts = np.array(y)
        return related_facts


    def split_rel(self, rel_idx, split_range, facts):
        facts1 = facts.copy()
        for i in range(len(split_range)):
            facts1[(facts1[:,1]==rel_idx) & np.isin(facts1[:,4] - facts1[:,3], split_range[i]), 1] = rel_idx + 100*i
        return facts1

    # Function to remove rows where the last five columns are the same as any preceding five columns
    def remove_matching_rows(self, df):
        n = df.shape[1]  # Number of columns
        last_five_cols = df.iloc[:, -5:]
        
        condition_to_remove = pd.Series([False] * len(df))
        
        # Iterate over each group of five columns, excluding the last five
        for i in range(0, n - 5, 4):
            comparison_cols = df.iloc[:, i:i + 5]
            condition_to_remove |= (last_five_cols.values == comparison_cols.values).all(axis=1)
        
        # Filter out the rows
        df_filtered = df[~condition_to_remove]
        
        # Reset the index
        df_filtered.reset_index(drop=True, inplace=True)
        
        return df_filtered


    def find_common_nodes(self, ls1, ls2):
        return [list(set(ls1[i]).intersection(set(ls2[i]))) for i in range(len(ls1))]


    def get_inv_idx(self, num_dataset, idx):
        if isinstance(idx, int):
            return idx - num_dataset if idx >= num_dataset else idx + num_dataset
        else:
            x = idx.copy()
            x[idx >= num_dataset] = x[idx >= num_dataset] - num_dataset
            x[idx < num_dataset] = x[idx < num_dataset] + num_dataset
            return x

    def get_inv_rel(self, rel_idx):
        if rel_idx < self.num_rel//2:
            return rel_idx + self.num_rel//2
        else:
            return rel_idx - self.num_rel//2

    def get_inv_rule(self, rule, ruleLen):
        x = rule[:ruleLen][::-1]
        return [self.get_inv_rel(u) for u in x] + rule[ruleLen:]

    def get_weights_savepath(self, rel_idx, ruleLen, train_idx):
        return self.weights_savepath + '_rel_'+ str(rel_idx) +'_len_'+ str(ruleLen) + '_idx_'+ str(train_idx) +'.json'

    def get_weights_savepath_v2(self, rel_idx):
        return self.weights_savepath + '_rel_'+ str(rel_idx) +'.json'


    def my_convert_to_list(self, res_prob_ls):
        new_res_prob_ls = []
        for prob_ls in res_prob_ls:
            new_prob_ls = []
            for prob in prob_ls:
                new_prob_ls.append(prob.tolist())
            new_res_prob_ls.append(new_prob_ls)
        return new_res_prob_ls

    def my_convert_to_list_v2(self, res_prob_ls):
        new_res_prob_ls = []
        for prob in res_prob_ls:
            new_res_prob_ls.append(prob.tolist())
        return new_res_prob_ls

    def my_merge_dict(self, dict1, dict2):
        for k in dict2.keys():
            if k not in dict1.keys():
                dict1[k] = dict2[k]
            else:
                dict1[k] += dict2[k]
        return dict1

    def my_split_list(self, ls1, max_num):
        n_s = len(ls1)//max_num
        res = []
        for i in range(n_s):
            res.append(ls1[max_num*i:max_num*(i+1)])

        if len(ls1)>max_num*n_s:
            res.append(ls1[max_num*n_s:])
        return res

    def obtain_edges_given_nodes(self, edges, st_node, end_node, query_int):
        x = (edges[:,0] == st_node) & (edges[:,2] == end_node)
        rel = edges[x, 1]
        tmp_rel = self.obtain_tmp_rel(edges[x, 3:], query_int)
        # print(edges[x,:])
        return rel, tmp_rel + 1


    def obtain_query_inv(self, query):
        x = int(self.num_rel/2)
        if query[1] > x:
            return [query[2], query[1]-x, query[0], query[3], query[4]]
        else:
            return [query[2], query[1]+x, query[0], query[3], query[4]]


    def collect_rules(self, idx_ls, rel_idx):
        '''
        Collect rules for shallow layers.
        '''
        if not isinstance(idx_ls, list):
            idx_ls = [idx_ls]

        rules_dict = []
        for idx in idx_ls:
            if self.overall_mode == 'general':
                path = '../output/found_paths/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path = '../output/found_paths_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'

            if not os.path.exists(path):
                # print(path + ' not found')
                cur_rules_dict = {}
            else:
                with open(path, 'r') as f:
                    cur_rules_dict = json.load(f)
            for s_i in cur_rules_dict.keys():
                rule_len = int(s_i)
                if self.f_non_Markovian:
                    mask_pre = [1]*(rule_len + int(rule_len>1)) if self.f_adjacent_TR_only else [1]*(2*rule_len + rule_len*(rule_len-1)/2)
                else:
                    mask_pre = [1]*2*rule_len
                alpha_dict = self.merge_rules(cur_rules_dict[s_i], mask_pre) # merged rules share the same shallow score
                rules_dict += [r for r in alpha_dict]

        # counter and select most freq
        self.shallow_rule_dict[rel_idx] = self.count_lists(rules_dict)[:self.shallow_score_length]

        return


    def merge_rules(self, rules_dict, mask_pre):
        '''
        When we merge rules, we consider masking them (ignore certain features in the rules).
        '''
        # mask_pre: [1, 1, 1, 0, 0, ...] decide which notations to preserve
        merged_rules_dict = {}
        for r in rules_dict:
            cur_rule = tuple([a*b for a, b in zip(r['rule'], mask_pre)])
            if cur_rule not in merged_rules_dict:
                merged_rules_dict[cur_rule] = []
            merged_rules_dict[cur_rule].append(r['alpha'])
        return merged_rules_dict


    def merge_dicts(self, dict_list, num_samples):
        merged_dict = defaultdict(list)
        for d in dict_list:
            merged_dict[tuple(d['rule'])].append(d['alpha'])
        merged_dict = dict(merged_dict)
        for rule in merged_dict:
            # print(merged_dict[rule], num_samples)
            merged_dict[rule] = np.mean(merged_dict[rule] + [0] * (num_samples - len(merged_dict[rule])))
        return merged_dict



    def data_collection(self, idx_ls, rel_idx):
        '''
        To process multiple samples at one time, we merge their rule dict as a whole matrix.
        Once we calculate the rule score with the matrix, we split the rules according to their sources.
        '''
        assert self.overall_mode in ['general', 'few', 'biased', 'time_shifting']
        idx_ls = [idx_ls] if not isinstance(idx_ls, list) else idx_ls

        rules_dict, rule_source_dict = {}, {}  # rule_source_dict: show the source of each rule
        shallow_rules_dict = []

        for (l, idx) in enumerate(idx_ls):
            path = '../output/found_paths/'+ self.dataset_using +'_train_query_'+str(idx)+'.json' if self.overall_mode == 'general' else \
                   '../output/found_paths_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
            # print(path)

            if not os.path.exists(path):
                # print(path + ' not found')
                cur_rules_dict = {}
            else:
                with open(path, 'r') as f:
                    cur_rules_dict = json.load(f)

            cur_shallow_rules_dict = []
            for rule_len in cur_rules_dict.keys():
                rules_dict[rule_len] = [] if rule_len not in rules_dict.keys() else rules_dict[rule_len]
                rules_dict[rule_len] += cur_rules_dict[rule_len]

                rule_source_dict[rule_len] = [] if rule_len not in rule_source_dict.keys() else rule_source_dict[rule_len]
                rule_source_dict[rule_len].append([l, len(cur_rules_dict[rule_len])])
                
                mask_pre = [1] * (2*int(rule_len) + int(rule_len>1)) if self.f_adjacent_TR_only else \
                           [1] * (2*int(rule_len) + int(rule_len)*(int(rule_len)-1)/2)
                merged_rules_dict = self.merge_rules(cur_rules_dict[rule_len], mask_pre)
                merged_rules_dict = [{'rule': r, 'alpha': np.mean(merged_rules_dict[r])} for r in merged_rules_dict if r in self.shallow_rule_dict[rel_idx]]
                cur_shallow_rules_dict += merged_rules_dict

            shallow_rules_dict.append(cur_shallow_rules_dict)    
            
        return rules_dict, rule_source_dict, shallow_rules_dict



    def create_training_idx_in_batch(self, i, num_queries, num_processes, rel_idx, train_edges):
        if rel_idx < self.num_rel//2:
            s = 0
        else:
            s = len(train_edges)//2

        n_t = len(train_edges)//2
        num_rest_queries = n_t - (i + 1) * num_queries
        if (num_rest_queries >= num_queries) and (i + 1 < num_processes):
            queries_idx = range(s+i*num_queries, s+(i+1)*num_queries)
        else:
            queries_idx = range(s+i*num_queries, s+n_t)

        return queries_idx


    def obtain_valid_train_idx(self, idx_range):
        valid_train_idx = []
        for idx in idx_range:
            if self.overall_mode == 'general':
                path = '../output/found_paths/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path = '../output/found_paths_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'

            if not os.path.exists(path):
                continue
            
            with open(path, 'r') as f:
                rules_dict = json.load(f)
            if len(rules_dict) == 0:
                continue
            
            valid_train_idx.append(idx)
        
        return valid_train_idx


    def save_weighs(self, targ_rel_ls, res_attn_rel_dict, res_attn_TR_dict, res_attn_TR_prime_dict, res_attn_ruleLen_dict, res_attn_shallow_score):
        for rel_idx in targ_rel_ls:
            if rel_idx not in res_attn_rel_dict:
                continue
            my_res = {}
            my_res['attn_rel_ls'] = self.my_convert_to_list(res_attn_rel_dict[rel_idx])
            my_res['attn_TR_ls'] = self.my_convert_to_list(res_attn_TR_dict[rel_idx])
            if self.f_non_Markovian:
                my_res['attn_TR_prime_ls'] = self.my_convert_to_list(res_attn_TR_prime_dict[rel_idx])
            my_res['attn_ruleLen'] = res_attn_ruleLen_dict[rel_idx].tolist()

            my_res['shallow_score'] = res_attn_shallow_score[rel_idx].tolist()
            my_res['shallow_rule_dict'] = self.shallow_rule_dict[rel_idx]

            if self.save_weights:
                cur_path = self.get_weights_savepath_v2(rel_idx)
                with open(cur_path, 'w') as f:
                    json.dump(my_res, f)


    def create_rule_source_matrix(self, rule_source, num_samples, num_rules_total):
        '''
        Create a one-hot matrix to show the source of each rule.
        '''
        # Initialize a zero matrix with the given shape
        one_hot_matrix = np.zeros((num_samples, num_rules_total), dtype=int)

        idx_start = 0
        # Fill in the one-hot matrix
        for idx, num_rules in rule_source:
            one_hot_matrix[idx, idx_start:idx_start+num_rules] = 1
            idx_start += num_rules

        # print(one_hot_matrix)
        return one_hot_matrix


    def prepare_inputs(self, idx_ls, const_pattern_ls, rel_idx, TR_ls_len):
        rules_dict, rule_source_dict, shallow_rules_dict = self.data_collection(idx_ls, rel_idx)
        
        input_dict = {}
        f_valid = 0
        shallow_rule_idx, shallow_rule_alpha = [], []
  
        for i in range(self.max_explore_len):
            rule_len = str(i+1)
            if rule_len in rules_dict.keys():
                rel_ls, TR_ls, alpha_ls = [], [], []
                for r in rules_dict[rule_len]:
                    if isinstance(r, dict):
                        rel_ls.append(r['rule'][:i+1])
                        TR_ls.append([const_pattern_ls.index(TR) for TR in r['rule'][i+1:]])
                        alpha_ls.append([r['alpha']])
                if len(rel_ls)>0:
                    input_dict[i] = {'rel': np.array(rel_ls), 'TR': np.array(TR_ls), 'alpha': np.array(alpha_ls), 
                                     'source': self.create_rule_source_matrix(rule_source_dict[rule_len], len(idx_ls), len(rel_ls))}  
                    f_valid = 1
            else:
                input_dict[i] = {'rel': np.zeros((1, i+1)), 'TR': np.zeros((1, TR_ls_len[i])), 'alpha': np.ones((1, 1)) * 1e-10, 
                                 'source': np.zeros((len(idx_ls), 1))}

        for sample in shallow_rules_dict:
            rule_idx = np.zeros((self.shallow_score_length,))
            rule_alpha = np.zeros((self.shallow_score_length,))
            for r in sample:
                cur_rule_idx = self.shallow_rule_dict[rel_idx].index(r['rule'])
                rule_idx[cur_rule_idx] = 1
                rule_alpha[cur_rule_idx] = r['alpha']
            
            shallow_rule_idx.append(rule_idx)
            shallow_rule_alpha.append(rule_alpha)
        
        shallow_rule_idx = np.array(shallow_rule_idx)
        shallow_rule_alpha = np.array(shallow_rule_alpha)
        
        return input_dict, f_valid, shallow_rule_idx, shallow_rule_alpha



    def prepare_inputs_tfm_test(self, train_edges, query_rel, query_time, candidate, p_rec, p_order, mu_pair, sigma_pair, lambda_pair):
        x = train_edges[train_edges[:,0] == candidate]

        h_rec = 1-p_rec[query_rel]

        related_rel_dict = []
        h_order = []
        h_pair = []
        for rel in np.unique(x[:,1]):
            if rel == query_rel:
                h_rec = p_rec[query_rel]

            if p_order[(query_rel, rel)]>0:
                related_rel_dict.append(rel)
                x1 = x[x[:,1] == rel][:,3] - query_time
                idx = np.argmin(np.abs(x1))
                if x1[idx] >=0:
                    h_order.append(p_order[(query_rel, rel)])
                else:
                    h_order.append(1-p_order[(query_rel, rel)])

                h_pair.append(norm(mu_pair[(query_rel, rel)], sigma_pair[(query_rel, rel)]).pdf(np.abs(x1[idx])))
                # h_pair.append(lambda_pair[(query_rel, rel)]*np.exp(-lambda_pair[(query_rel, rel)]*np.abs(x1[idx])))

        return related_rel_dict, h_rec, h_order, h_pair


    def prepare_inputs_tfm_Wc_test(self, query_rel, ts_dict, p_order, mu_pair, sigma_pair, lambda_pair):
        related_rel_dict = []
        h_order = []
        h_pair = []
        for rel in ts_dict:
            if p_order[(query_rel, rel)]>0:
                related_rel_dict.append(rel)
                x1 = ts_dict[rel]
                idx = np.argmin(np.abs(x1))
                if x1[idx] >=0:
                    h_order.append(p_order[(query_rel, rel)])
                else:
                    h_order.append(1-p_order[(query_rel, rel)])

                h_pair.append(norm(mu_pair[(query_rel, rel)], sigma_pair[(query_rel, rel)]).pdf(np.abs(x1[idx])))
                # h_pair.append(lambda_pair[(query_rel, rel)]*np.exp(-lambda_pair[(query_rel, rel)]*np.abs(x1[idx])))

        return related_rel_dict, h_order, h_pair



    def prepare_inputs_tfm(self, train_edges, idx, p_rec, p_order, mu_pair, sigma_pair, lambda_pair):
        query = train_edges[idx]
        if train_edges[idx][1]>= self.num_rel//2:
            query_rel = train_edges[idx][1] - self.num_rel//2
        else:
            query_rel = train_edges[idx][1] + self.num_rel//2

        query_rel_one_hot = [0] * self.num_rel
        query_rel_one_hot[query_rel] = 1

        h_rec = 1-p_rec[query_rel]

        rev_query = [query[2], query_rel, query[0], query[3], query[4]]
        x = train_edges[~np.all(train_edges == rev_query, axis=1)]
        x = x[x[:,0] == rev_query[0]]

        related_rel_dict = []
        h_order = []
        h_pair = []
        for rel in np.unique(x[:,1]):
            if rel == query_rel:
                h_rec = p_rec[query_rel]

            if p_order[(query_rel, rel)]>0:
                related_rel_dict.append(rel)
                x1 = x[x[:,1] == rel][:,3] - rev_query[3]
                idx = np.argmin(np.abs(x1))
                if x1[idx] >=0:
                    h_order.append(p_order[(query_rel, rel)])
                else:
                    h_order.append(1-p_order[(query_rel, rel)])

                h_pair.append(norm(mu_pair[(query_rel, rel)], sigma_pair[(query_rel, rel)]).pdf(np.abs(x1[idx])))
                # h_pair.append(lambda_pair[(query_rel, rel)]*np.exp(-lambda_pair[(query_rel, rel)]*np.abs(x1[idx])))

        if len(related_rel_dict)==0:
            f_exist = [1,0,0]
            related_rel_dict = [0]
            h_order = [0]
            h_pair = [0]
        else:
            f_exist = [1,1,1]

        input_dict = {}
        input_dict['query_rel'] = np.array([query_rel])
        input_dict['query_rel_one_hot'] = np.array(query_rel_one_hot)
        input_dict['related_rel_dict'] = np.array(related_rel_dict)
        input_dict['h_rec'] = np.array([h_rec])
        input_dict['h_order'] = np.array(h_order)
        input_dict['h_pair'] = np.array(h_pair)
        input_dict['f_exist'] = np.array(f_exist)

        # print(input_dict)

        return input_dict


    def prepare_inputs_tfm_v2(self, train_edges, idx_ls, p_rec, p_order, mu_pair, sigma_pair, lambda_pair):
        query_rel = []
        query_rel_one_hot = []
        related_rel_dict = []
        h_rec = []
        h_order = []
        h_pair = []
        f_exist = []
        mask = []

        max_num_related_rel = self.num_rel
        for idx in idx_ls:
            cur_query = train_edges[idx]
            if train_edges[idx][1]>= self.num_rel//2:
                cur_query_rel = train_edges[idx][1] - self.num_rel//2
            else:
                cur_query_rel = train_edges[idx][1] + self.num_rel//2

            cur_query_rel_one_hot = [0] * self.num_rel
            cur_query_rel_one_hot[cur_query_rel] = 1

            cur_h_rec = 1-p_rec[cur_query_rel]

            cur_rev_query = [cur_query[2], cur_query_rel, cur_query[0], cur_query[3], cur_query[4]]
            x = train_edges[~np.all(train_edges == cur_rev_query, axis=1)]
            x = x[x[:,0] == cur_rev_query[0]]

            cur_related_rel_dict = []
            cur_h_order = []
            cur_h_pair = []
            for rel in np.unique(x[:,1]):
                if rel == cur_query_rel:
                    cur_h_rec = p_rec[cur_query_rel]

                if p_order[(cur_query_rel, rel)]>0:
                    cur_related_rel_dict.append(rel)
                    x1 = x[x[:,1] == rel][:,3] - cur_rev_query[3]
                    idx = np.argmin(np.abs(x1))
                    if x1[idx] >=0:
                        cur_h_order.append(p_order[(cur_query_rel, rel)])
                    else:
                        cur_h_order.append(1-p_order[(cur_query_rel, rel)])

                    cur_h_pair.append(norm(mu_pair[(cur_query_rel, rel)], sigma_pair[(cur_query_rel, rel)]).pdf(np.abs(x1[idx])))
                    # cur_h_pair.append(lambda_pair[(cur_query_rel, rel)]*np.exp(-lambda_pair[(cur_query_rel, rel)]*np.abs(x1[idx])))

            mask.append([1.]*len(cur_related_rel_dict))

            if len(cur_related_rel_dict)==0:
                cur_f_exist = [1,0,0]
                cur_related_rel_dict = [0]
                cur_h_order = [0]
                cur_h_pair = [0]
            else:
                cur_f_exist = [1,1,1]

            query_rel.append([cur_query_rel])
            query_rel_one_hot.append(cur_query_rel_one_hot)
            related_rel_dict.append(cur_related_rel_dict)
            h_rec.append([cur_h_rec])
            h_order.append(cur_h_order)
            h_pair.append(cur_h_pair)
            f_exist.append(cur_f_exist)
            # max_num_related_rel = max(max_num_related_rel, len(cur_related_rel_dict))

        for i in range(len(idx_ls)):
            related_rel_dict[i] += [0]*(max_num_related_rel-len(related_rel_dict[i]))
            h_order[i] += [0]*(max_num_related_rel-len(h_order[i]))
            h_pair[i] += [0]*(max_num_related_rel-len(h_pair[i]))
            mask[i] += [0]*(max_num_related_rel-len(mask[i]))

        input_dict = {}
        input_dict['query_rel'] = np.array(query_rel)
        input_dict['query_rel_one_hot'] = np.array(query_rel_one_hot)
        input_dict['related_rel_dict'] = np.array(related_rel_dict)
        input_dict['h_rec'] = np.array(h_rec)
        input_dict['h_order'] = np.array(h_order)
        input_dict['h_pair'] = np.array(h_pair)
        input_dict['f_exist'] = np.array(f_exist)
        input_dict['mask'] = np.array(mask)

        # print(input_dict)

        return input_dict


    def prepare_inputs_tfm_Wc_v2(self, train_edges, idx_ls, p_order, mu_pair, sigma_pair, lambda_pair):
        query_rel_one_hot = []
        related_rel_dict = []
        h_order = []
        h_pair = []
        f_exist = []
        mask = []

        max_num_related_rel = self.num_rel
        for idx in idx_ls:
            cur_query_rel = train_edges[idx][1]
            cur_query_rel_one_hot = [0] * self.num_rel
            cur_query_rel_one_hot[cur_query_rel] = 1

            cur_related_rel_dict = []
            cur_h_order = []
            cur_h_pair = []

            if self.overall_mode == 'general':
                path = '../output/found_time_gaps/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path = '../output/found_time_gaps_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json'

            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)

                for k in data.keys():
                    if isinstance(data[k], int):
                        rel = int(k)

                        if p_order[(cur_query_rel, rel)]>0:
                            cur_related_rel_dict.append(rel)
                            if data[k] >=0:
                                cur_h_order.append(p_order[(cur_query_rel, rel)])
                            else:
                                cur_h_order.append(1-p_order[(cur_query_rel, rel)])

                            cur_h_pair.append(norm(mu_pair[(cur_query_rel, rel)], sigma_pair[(cur_query_rel, rel)]).pdf(np.abs(data[k])))
                            # cur_h_pair.append(lambda_pair[(cur_query_rel, rel)]*np.exp(-lambda_pair[(cur_query_rel, rel)]*np.abs(data[k])))

            mask.append([1.]*len(cur_related_rel_dict))

            if len(cur_related_rel_dict)==0:
                cur_f_exist = [0,0]
                cur_related_rel_dict = [0]
                cur_h_order = [0]
                cur_h_pair = [0]
            else:
                cur_f_exist = [1,1]

            query_rel_one_hot.append(cur_query_rel_one_hot)
            related_rel_dict.append(cur_related_rel_dict)
            h_order.append(cur_h_order)
            h_pair.append(cur_h_pair)
            f_exist.append(cur_f_exist)

        for i in range(len(idx_ls)):
            related_rel_dict[i] += [0]*(max_num_related_rel-len(related_rel_dict[i]))
            h_order[i] += [0]*(max_num_related_rel-len(h_order[i]))
            h_pair[i] += [0]*(max_num_related_rel-len(h_pair[i]))
            mask[i] += [0]*(max_num_related_rel-len(mask[i]))

        input_dict = {}
        input_dict['query_rel_one_hot'] = np.array(query_rel_one_hot)
        input_dict['related_rel_dict'] = np.array(related_rel_dict)
        input_dict['h_order'] = np.array(h_order)
        input_dict['h_pair'] = np.array(h_pair)
        input_dict['f_exist'] = np.array(f_exist)
        input_dict['mask'] = np.array(mask)

        return input_dict