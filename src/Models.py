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




class TILP(object):
    def __init__(self, num_rel, num_pattern, num_ruleLen, num_paths_dict, dataset_using, overall_mode):
        self.num_rel = num_rel
        self.num_pattern = num_pattern
        self.num_ruleLen = num_ruleLen
        self.dataset_using = dataset_using

        if self.dataset_using == 'wiki':
            self.num_entites = 12554
            self.prob_cal_alpha = 1  # do sampling when calculating arriving rate for training
        elif self.dataset_using == 'YAGO':
            self.num_entites = 10623
            self.prob_cal_alpha = 1

        self.num_train_samples_max = 100000
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
        f_non_Markovian: TR(I_1, I_q), TR(I_2, I_q), ..., TR(I_N, I_q), TR(I_1, I_2), TR(I_1, I_3), ..., TR(I_{N-1}, I_N)
        f_non_Markovian and f_adjacent_TR_only: TR(I_1, I_q), TR(I_N, I_q), TR(I_1, I_2), TR(I_2, I_3), ..., TR(I_{N-1}, I_N)
        '''
        self.f_non_Markovian = True # consider non-Markovian constraints
        self.f_adjacent_TR_only = True # consider adjacent TRs only

        self.f_Wc_ts = False # consider intermediate nodes for temporal feature modeling
        self.max_rulenum = {1: 20, 2: 50, 3: 100, 4: 100, 5: 200}

        # add shallow layers to enhance expressiveness
        self.gamma_shallow = 0.2
        self.shallow_score_length = 400

        if self.overall_mode == 'general':
            self.weights_savepath = '../output/train_weights/train_weights_' + self.dataset_using
        elif self.overall_mode in ['few', 'biased', 'time_shifting']:
            self.weights_savepath = '../output/train_weights_'+ self.overall_mode +'/train_weights_'\
                                     + self.dataset_using


    def obtain_valid_train_idx(self, idx_range):
        valid_train_idx = []
        for idx in idx_range:
            if self.overall_mode == 'general':
                path = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path = '../output/found_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'

            if not os.path.exists(path):
                continue
            valid_train_idx.append(idx)
        return valid_train_idx


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
        query_score_shallow, var_attn_shallow_score = self.build_shallow_score()
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
                            cur_input_dict[rel_dict[l]] = x[l]['rel']
                            cur_input_dict[TR_dict[l]] = x[l]['TR']
                            cur_input_dict[alpha_dict[l]] = x[l]['alpha']
                            # print(x[l])

                        cur_input_dict[self.rel_idx] = [rel_idx]
                        cur_input_dict[self.shallow_rule_idx] = shallow_rule_idx
                        cur_input_dict[self.shallow_rule_alpha] = shallow_rule_alpha

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



    def train_tfm_v2(self, rel_idx_ls, num_training, train_edges, dist_pars):
        p_rec, p_order, mu_pair, sigma_pair, lambda_pair = dist_pars
        var_W_rec, var_b_rec, var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm = self.variable_init_tfm()

        query_rel = tf.placeholder(tf.int64, shape=(None, 1))
        query_rel_one_hot = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        related_rel_dict = tf.placeholder(tf.int64, shape=(None, self.num_rel))
        h_rec = tf.placeholder(tf.float32, shape=(None, 1))
        h_order = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        h_pair = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        f_exist = tf.placeholder(tf.float32, shape=(None, 3))
        mask = tf.placeholder(tf.float32, shape=(None, self.num_rel))


        valid_train_idx = []
        for idx in range(len(train_edges)):
            if self.overall_mode == 'general':
                path = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path = '../output/found_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'

            if not os.path.exists(path):
                continue
            valid_train_idx.append(idx)


        query_score = self.calculate_tfm_score_v2(query_rel, query_rel_one_hot, related_rel_dict, h_rec, h_order, h_pair, 
                                              f_exist, mask, var_W_rec, var_b_rec,
                                              var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm)
        final_loss = -tf.math.log(query_score)

        optimizer = tf.train.AdamOptimizer(0.001)
        gvs = optimizer.compute_gradients(final_loss)
        optimizer_step = optimizer.apply_gradients(gvs)

        feed_list = [var_W_rec, var_b_rec, var_W_order_ls, var_b_order_ls, var_W_pair_ls, \
                                    var_b_pair_ls, gamma_tfm, final_loss, optimizer_step]
        loss_avg_old = 100

        init = tf.global_variables_initializer()

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


    def train_tfm_Wc_v2(self, rel_idx_ls, num_training, train_edges, dist_pars):
        p_order, mu_pair, sigma_pair, lambda_pair = dist_pars
        var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm = self.variable_init_tfm_Wc()

        query_rel_one_hot = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        related_rel_dict = tf.placeholder(tf.int64, shape=(None, self.num_rel))
        h_order = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        h_pair = tf.placeholder(tf.float32, shape=(None, self.num_rel))
        f_exist = tf.placeholder(tf.float32, shape=(None, 2))
        mask = tf.placeholder(tf.float32, shape=(None, self.num_rel))


        valid_train_idx = []
        for idx in range(len(train_edges)):
            if self.overall_mode == 'general':
                path = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path = '../output/found_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'

            if not os.path.exists(path):
                continue
            valid_train_idx.append(idx)


        query_score = self.calculate_tfm_Wc_score_v2(query_rel_one_hot, related_rel_dict, h_order, h_pair, f_exist, mask,
                                                     var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm)
        final_loss = -tf.math.log(query_score)

        optimizer = tf.train.AdamOptimizer(0.001)
        gvs = optimizer.compute_gradients(final_loss)
        optimizer_step = optimizer.apply_gradients(gvs)

        feed_list = [var_W_order_ls, var_b_order_ls, var_W_pair_ls, var_b_pair_ls, gamma_tfm, final_loss, optimizer_step]
        loss_avg_old = 100

        init = tf.global_variables_initializer()

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


    def apply_single_rule(self, start_node, rule, rule_Len, facts, 
                                f_print=0, return_walk=False, targ_node=None, 
                                return_edges=False):
        rel_ls = rule[:rule_Len] # [r_1, r_2, ..., r_N]
        TR_ls = rule[rule_Len:]  # [TR(I_1, I_q), TR(I_N, I_q), TR(I_1, I_2), ...] if only adjacent TRs are considered else [TR(I_1, I_q), TR(I_2, I_q), ..., TR(I_N, I_q), TR(I_1, I_2), ...]

        path_ls = []
        cur_nodes = [int(start_node)]
        facts = facts.astype(int) # [s, r, o, TR(I, I_q), ts, te]

        for i in range(rule_Len):
            if self.f_non_Markovian and self.f_adjacent_TR_only and i != 0 and i != rule_Len - 1:
                cur_edges = facts[np.isin(facts[:, 0], cur_nodes) & (facts[:, 1] == rel_ls[i])]
            else:
                # if only adjacent TRs are considered
                cur_edges = facts[np.isin(facts[:, 0], cur_nodes) & (facts[:, 1] == rel_ls[i]) & (facts[:, 3] == TR_ls[min(i, 1)])]
                # cur_edges = facts[np.isin(facts[:, 0], cur_nodes) & (facts[:, 1] == rel_ls[i]) & (facts[:, 3] == TR_ls[i])]

            # print(cur_edges.shape)
            # print(cur_edges)
            # print('---------------------')

            if len(cur_edges) == 0:
                if return_edges:
                    return []
                if return_walk:
                    return {}, []
                else:
                    return {}
    
            cur_nodes = cur_edges[:, 2]
            path_ls.append(cur_edges[:,[0,1,4,5,2]])


        if isinstance(targ_node, int) or isinstance(targ_node, float):
            path_ls[-1] = path_ls[-1][path_ls[-1][:,-1] == targ_node]

        if return_edges:
            return path_ls
       
        heads = ['entity_', 'rel_', 'ts_', 'te_'] # we need to remove path that contains repeated edges
        cur_ent_walk_res = self.get_walks(path_ls, heads, rule, rule_Len).to_numpy()
        
        path_ls = []
        cur_nodes = cur_ent_walk_res[:,-1:]
        if f_print:
            print(cur_ent_walk_res)

        if not return_walk:
            cur_ent_walk_res = 0

        df = pd.DataFrame(cur_nodes, columns=["end_node"], dtype=int)
        res = df["end_node"].value_counts(normalize=True).to_dict()


        if return_walk:
            return res, cur_ent_walk_res
        return res


    def alpha_calculation(self, path1, masked_facts, rel_idx, query, time_shift_mode):
        rule_dict = {}
        if not os.path.exists(path1):
            return rule_dict
        with open(path1, 'r') as f:
            rule_dict_loaded = json.load(f)
        
        if self.overall_mode == 'general':
            with open('../output/learned_rules/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json','r') as f:
                rule_dict_summarized = json.load(f)
        elif self.overall_mode in ['few', 'biased', 'time_shifting']:
            with open('../output/learned_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json','r') as f:
                rule_dict_summarized = json.load(f)

        TR = self.obtain_tmp_rel(masked_facts[:, 3:], query[3:]).reshape((-1,1))
        masked_facts_with_TR = np.hstack((masked_facts[:, :3], TR)) if not self.f_non_Markovian else np.hstack((masked_facts[:, :3], TR, masked_facts[:, 3:]))
        masked_facts_with_TR = np.unique(masked_facts_with_TR, axis=0)                

        for num in rule_dict_loaded:
            rules = [r_dict['rule'] for r_dict in rule_dict_loaded[num]]
            for r in rules:
                if r not in [r_dict['rule'] for r_dict in rule_dict_summarized[num]]:
                    continue
                if time_shift_mode in [-1, 1] and time_shift_mode in r[int(num):int(num) + 2]:
                    continue
                # print(r)
                # Instead of calculating alpha for all rules, we select most frequent ones
                rand_num = random.random()
                if rand_num < self.prob_cal_alpha:
                    cur_dict = self.apply_single_rule(query[0], r, int(num), masked_facts_with_TR)
                    # print(cur_dict)
                    # alpha = 0.01 # if we use sampling, it's possible we cannot find the targ node
                    if query[2] in cur_dict.keys():
                        alpha = cur_dict[query[2]]
                else:
                    alpha = 'Unknown'

                if num not in rule_dict:
                    rule_dict[num] = []
                rule_dict[num].append({'rule': r, 'alpha': alpha})
        return rule_dict


    def path_search(self, path2, masked_facts, query, time_shift_mode):
        '''
        Different TR setting:
        f_Markovian: TR(I_1, I_q), TR(I_2, I_q), ..., TR(I_N, I_q)
        f_non_Markovian: TR(I_1, I_q), TR(I_2, I_q), ..., TR(I_N, I_q), TR(I_1, I_2), TR(I_1, I_3), ..., TR(I_{N-1}, I_N)
        f_non_Markovian and f_adjacent_TR_only: TR(I_1, I_q), TR(I_N, I_q), TR(I_1, I_2), TR(I_2, I_3), ..., TR(I_{N-1}, I_N)
        '''
        if self.f_Wc_ts:
            t_s_dict = {}
            for k1 in range(self.num_rel):
                t_s_dict[k1] = []

        rule_dict = {}
        edges_simp = masked_facts[:,[0,2]]
        edges_simp = np.unique(edges_simp, axis=0)
        edges_simp = edges_simp.astype(int)
        pos = list(edges_simp)
        rows, cols = zip(*pos)

        adj_mat = np.zeros((self.num_entites, self.num_entites))
        adj_mat[rows, cols] = 1

        cur_num_hops1, new_nodes_ls1 = self.BFS_mat_ver2(query[0], adj_mat, self.num_entites, query[2], self.max_explore_len)
        if len(cur_num_hops1) > 0:
            _, new_nodes_ls2 = self.BFS_mat_ver2(query[2], adj_mat, self.num_entites, query[0], self.max_explore_len)
            for num in cur_num_hops1:
                path_ls = self.find_common_nodes(new_nodes_ls1[:num+1], new_nodes_ls2[:num+1][::-1])
                walk_edges = []
                for i in range(num):
                    related_facts = masked_facts[np.isin(masked_facts[:,0], path_ls[i]) & np.isin(masked_facts[:,2], path_ls[i+1])]
                    walk_edges.append(related_facts[:,[0,1,3,4,2]]) # [s, r, ts, te, o]

                    if self.f_Wc_ts:
                        z = masked_facts[np.isin(masked_facts[:,0], path_ls[i]) & np.isin(masked_facts[:,2], path_ls[i+1])]
                        for z1 in z:
                            t_s_dict[z1[1]].append(z1[3]-query[3])

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
                    # TR(I_i, I_j)
                    for i in range(num-1):
                        for j in range(i+1, num):
                            if self.f_adjacent_TR_only and j != i + 1:
                                # instead of considering all TR, we preserve adjacent TR: TR(I_i, I_i+1)
                                continue
                            cur_TR = self.obtain_tmp_rel(cur_ent_walk_res[:, 4*i+2:4*i+4], cur_ent_walk_res[:, 4*j+2:4*j+4]).reshape((-1,1))
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

            if self.f_Wc_ts:
                for k1 in t_s_dict.keys():   
                    t_s_dict[k1] = t_s_dict[k1][np.argmin(np.abs(t_s_dict[k1]))] if len(t_s_dict[k1])>0 else None
                with open(path2, 'w') as f:
                    json.dump(t_s_dict, f)
        return rule_dict


    def apply(self, train_edges, rel_idx=None, idx_ls=None, pos_examples_idx=None, time_shift_mode=0, mode='path_search'):
        '''
        mode: 'path_search' or 'alpha_calculation'
        1) path_search: given a query, find all paths that satisfy the query
        2) alpha_calculation: given a query, calculate alpha for all rules
        '''

        idx_ls1 = idx_ls if idx_ls is not None else range(len(train_edges))

        for idx in idx_ls1:
            if self.overall_mode == 'general':
                path1 = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json'
                path2 = '../output/found_t_s/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path1 = '../output/found_rules_'+ self.overall_mode +'/'+ self.dataset_using + '_train_query_'+str(idx)+'.json'
                path2 = '../output/found_t_s_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json'

            query = train_edges[idx]
            # print(query, idx)

            if isinstance(rel_idx, int) and (query[1] != rel_idx):
                continue

            if (pos_examples_idx is not None) and (idx not in pos_examples_idx):
                continue

            # mask ori query and inv query in the train set
            masked_facts = np.delete(train_edges, [idx, self.get_inv_idx(len(train_edges)//2, idx)], 0)
            masked_facts = masked_facts[masked_facts[:,3]<=query[3]] if self.overall_mode == 'time_shifting' else masked_facts
            
            if mode == 'alpha_calculation':
                rule_dict = self.alpha_calculation(path1, masked_facts, rel_idx, query, time_shift_mode)
            elif mode == 'path_search':
                rule_dict = self.path_search(path2, masked_facts, query, time_shift_mode)
                
            with open(path1, 'w') as f:
                json.dump(rule_dict, f)

        return 



    def apply_in_batch(self, i, num_queries, num_processes, rel_idx, mode, train_edges, path_name='', 
                            pos_examples_idx=None, time_shift_mode=0):
        queries_idx = self.create_training_idx_in_batch(i, num_queries, num_processes, rel_idx, train_edges)
        self.apply(train_edges, rel_idx, queries_idx, pos_examples_idx, time_shift_mode, mode=mode)

        return 



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



    def predict(self, rel_idx, bg_pred, test_data, test_data_inv, const_pattern_ls, assiting_data, dist_pars, train_edges,
                    queries_idx = None, tol_gap=[0, 0], selected_rules=None, enable_pure_guessing=True,
                    format_extra_len=0, f_predicting=0):

        if self.overall_mode == 'general':
            f_name = 'learned_rules'
        elif self.overall_mode in ['few', 'biased', 'time_shifting']:
            f_name = 'learned_rules_' + self.overall_mode
        path = '../output/'+f_name+'/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json'

        if not os.path.exists(path):
            # print(path + ' not found')
            rule_dict1 = {}
            if not enable_pure_guessing:
                return {}
        else:
            with open(path,'r') as f:
                rule_dict1 = json.load(f)

        if rel_idx < self.num_rel//2:
            data_using = test_data
            s = 0
        else:
            data_using = test_data_inv
            s = len(test_data)

        rule_score_bar = {}
        for rule_Len in rule_dict1.keys():
            r_score_ls = []
            for r in rule_dict1[rule_Len]:
                r_score_ls.append(r['score'])
            r_score_ls.sort(reverse=True)
            rule_score_bar[rule_Len] = r_score_ls[min(len(r_score_ls)-1, self.max_rulenum[int(rule_Len)])]


        mode = 0
        if self.f_Wc_ts or self.f_non_Markovian:
            mode = 1

        if not queries_idx:
            queries_idx = range(len(data_using))
        

        rank_dict = {}
        for idx in queries_idx:
            if data_using[idx][1] != rel_idx:
                continue
            
            line = data_using[idx]
            cur_bg_pred = bg_pred[bg_pred[:, 3]<line[3]] if f_predicting else bg_pred
    
            res_dict, res_ts_dict = {}, {}
            x = self.create_mapping_facts(cur_bg_pred, line[3:], mode)
            for rule_Len in rule_dict1.keys():
                for r in rule_dict1[rule_Len]:
                    if isinstance(selected_rules, dict):
                        if not r['rule'] in selected_rules[rel_idx]:
                            continue
                    if r['score'] < rule_score_bar[rule_Len]:
                        continue

                    cur_dict, cur_walk = self.apply_single_rule(line[0], r['rule'], int(rule_Len), 
                                                                x, f_print=0, return_walk=True)

                    if self.f_Wc_ts:
                        for w in cur_walk:
                            if w[-1] not in res_ts_dict:
                                res_ts_dict[w[-1]] = {}
                            for l in range(int(rule_Len)):
                                if r['rule'][l] not in res_ts_dict[w[-1]]:
                                    res_ts_dict[w[-1]][r['rule'][l]] = []
                                res_ts_dict[w[-1]][r['rule'][l]].append(w[3*l+1]-line[3])

                    for k in cur_dict.keys():
                        cur_dict[k] = cur_dict[k] * r['score']

                    res_dict = self.my_merge_dict(res_dict, cur_dict)

            if enable_pure_guessing and (len(res_dict) == 0):
                for i in range(self.num_entites):
                    res_dict[i] = np.random.normal(loc=0.0, scale=0.01, size=None)

            res_tfm_score = self.predict_tfm_score_with_fact_check(line, bg_pred, train_edges, res_dict, dist_pars, assiting_data, format_extra_len)

            rank = self.evaluate_res_dict(res_dict, line[2], line, train_edges, res_tfm_score, tol_gap)
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


        f_check_enable = [0, 0.5, 1][0] # whether enable

        if self.dataset_using in ['wiki', 'YAGO']:
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


    def _random_uniform_unit(self, r, c):
        bound = 6./ np.sqrt(c)
        init_matrix = np.random.uniform(-bound, bound, (r, c))
        init_matrix = np.array(map(lambda row: row / np.linalg.norm(row), init_matrix))
        return init_matrix


    def build_input(self):
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


    def build_shallow_score(self):
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



    def build_rnn_graph(self):
        rnn_inputs_ls = self.build_input()

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
                for j in range(i-1):
                    for k in range(j+1,i):
                        if self.f_adjacent_TR_only and k != j+1:
                            # for f_adjacent_TR_only, we do not create the variables
                            continue
                        x = tf.stack([rnn_outputs[j], rnn_outputs[k]], axis=1)
                        x = tf.reshape(x, (self.rnn_batch_size, -1))
                        x = tf.nn.softmax(tf.matmul(x, self.W_TR_prime) + self.b_TR_prime)
                        cur_attn_TR_prime_ls.append(tf.reshape(x[0], [-1,1]))

                attn_TR_prime_ls.append(cur_attn_TR_prime_ls)


        return attn_rel_ls, attn_TR_ls, attn_TR_prime_ls, attn_ruleLen




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



    def calculate_query_score(self, edges, ent_walk_res, query_int, var_prob_rel_ls, var_prob_pattern_ls):
        path_len = ent_walk_res.shape[1]-1
        prob_path_ls = []
        for l in range(ent_walk_res.shape[0]):
            cur_prob_path = tf.constant(1.)
            for k in range(path_len):
                rel, tmp_rel = self.obtain_edges_given_nodes(edges, ent_walk_res[l,k], ent_walk_res[l,k+1], query_int)

                prob_rel = tf.nn.embedding_lookup(var_prob_rel_ls[path_len-1][k], rel)
                prob_pattern = tf.nn.embedding_lookup(var_prob_pattern_ls[path_len-1][k], tmp_rel)
                cur_prob_path = cur_prob_path * tf.reduce_sum(tf.nn.softmax(prob_rel * prob_pattern) * prob_rel * prob_pattern, 0)

            prob_path_ls.append(cur_prob_path)

        prob_path = tf.concat(prob_path_ls, 0)
        prob_query = tf.reduce_sum(tf.nn.softmax(prob_path) * prob_path, 0)

        return prob_query



    def calculate_query_score_fun_ver2(self, rel_dict, tmp_rel_dict, rPro_dict, var_prob_rel_ls, var_prob_pattern_ls, var_prob_ruleLen):
        prob = tf.constant(0.)
        for l in range(self.max_explore_len):
            prob_rel = tf.constant(1.)
            # print(l)
            for k in range(l+1):
                prob_rel = prob_rel * tf.nn.embedding_lookup(var_prob_rel_ls[l][k], rel_dict[l][:,k])
            # print(prob_rel.shape)

            prob_pattern = tf.constant(1.)
            for k in range((l+1)*(l+2)//2):
                prob_pattern = prob_pattern * tf.nn.embedding_lookup(var_prob_pattern_ls[l][k], tmp_rel_dict[l][:,k])
            # print(prob_pattern.shape)

            cur_prob = prob_rel * prob_pattern * rPro_dict[l]
            # print(cur_prob.shape)

            cur_prob = tf.reduce_sum(cur_prob, 0) * var_prob_ruleLen[0, l]
            # print(cur_prob.shape)

            prob += cur_prob

        return prob


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


    def create_mapping_facts(self, facts, query_int, mode=0):
        x = self.obtain_tmp_rel(facts[:, 3:], query_int).reshape((-1,1))
        if mode:
            x = np.hstack((facts[:, :3], x, facts[:, 3:]))
        else:
            x = np.hstack((facts[:, :3], x))
            x = np.unique(x, axis=0)
            x = x.astype(int)
        return x


    def write_all_rule_dicts(self, rel_idx, rule_dict, rule_sup_num_dict, const_pattern_ls, cal_score=False, select_topK_rules=True):
        if cal_score:
            # calculate rule weights
            cur_path = self.get_weights_savepath_v2(rel_idx)

            if not os.path.exists(cur_path):
                # print(cur_path + ' not found')
                return
            with open(cur_path, 'r') as f:
                my_res = json.load(f)

            var_prob_rel_ls = [np.squeeze(np.array(x), 2) for x in my_res['attn_rel_ls']]
            var_prob_pattern_ls = [np.squeeze(np.array(x), 2) for x in my_res['attn_TR_ls']]
            var_prob_pattern_prime_ls = [[]] + [np.squeeze(np.array(x), 2) for x in my_res['attn_TR_prime_ls'][1:]] if self.f_non_Markovian else []
            var_prob_ruleLen = np.array(my_res['attn_ruleLen'])


        rule_dict1 = {}
        for rule_Len in rule_dict.keys():
            rule_dict1[rule_Len] = []
            for r in rule_dict[rule_Len]:
                r = r.tolist()
                cur_rule_dict = {'rule': r, 'sup_num': rule_sup_num_dict[str(r)]}

                if cal_score:
                    s = self.calculate_rule_score(r, int(rule_Len), const_pattern_ls, var_prob_rel_ls, var_prob_pattern_ls, 
                                                    var_prob_pattern_prime_ls, var_prob_ruleLen)
                    s *= (1-self.gamma_shallow)
                    s += (self.gamma_shallow) * my_res['shallow_score'][my_res['shallow_rule_dict'].index(r)] if r in my_res['shallow_rule_dict'] else 0
                    cur_rule_dict['score'] = s

                rule_dict1[rule_Len].append(cur_rule_dict)
                
            if select_topK_rules:
                rule_dict1[rule_Len] = sorted(rule_dict1[rule_Len], key=lambda x: x['sup_num'], reverse=True)[:self.max_rulenum[int(rule_Len)]]


        if self.overall_mode == 'general':
            with open('../output/learned_rules/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json','w') as f:
                json.dump(rule_dict1, f)
        elif self.overall_mode in ['few', 'biased', 'time_shifting']:
            with open('../output/learned_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json','w') as f:
                json.dump(rule_dict1, f)

        return 


    def create_all_rule_dicts(self, rel_idx, train_edges, query_idx=None, pos_examples_idx=None):
        rule_dict = {}
        rule_sup_num_dict = {}
        cnt = 0

        idx_ls = range(len(train_edges))

        if not query_idx:
            query_idx = range(len(train_edges))

        for idx1 in query_idx:
            idx = idx_ls[idx1]
            if isinstance(pos_examples_idx, list) and idx not in pos_examples_idx:
                continue
            if train_edges[idx][1] == rel_idx:
                if self.overall_mode == 'general':
                    path = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
                elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                    path = '../output/found_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'

                if not os.path.exists(path):
                    # print(path + ' not found')
                    continue
                with open(path, 'r') as f:
                    x = json.load(f)
                cnt += 1
                for k in x.keys():
                    if int(k)<=self.max_explore_len:
                        if len(x[k]) >1:
                            cur_rule_mat = np.vstack([r['rule'] for r in x[k]])
                        else:
                            cur_rule_mat = np.array([r['rule'] for r in x[k]])
                        cur_rule_mat = np.unique(cur_rule_mat, axis=0)

                        rule_sup_num_dict = self.my_merge_dict(rule_sup_num_dict, dict(Counter([str(r['rule']) for r in x[k]])))

                        if k not in rule_dict.keys():
                            rule_dict[k] = cur_rule_mat.copy()
                        else:
                            rule_dict[k] = np.vstack((rule_dict[k], cur_rule_mat))
                        rule_dict[k] = np.unique(rule_dict[k], axis=0)

        return rule_dict, rule_sup_num_dict



    def create_rule_supplement(self, rel_idx, train_edges, query_idx=None):
        with open('../output/learned_rules/'+ self.dataset_using +'_all_rules_'+str(rel_idx)+'.json','r') as f:
            rule_dict1 = json.load(f)

        with open('../output/'+ self.dataset_using +'_contri_dict_'+str(rel_idx)+'.json','r') as f:
            contri_dict = json.load(f)

        if not query_idx:
            query_idx = range(len(train_edges))

        rule_sup = {}
        for rule_Len in rule_dict1.keys():
            if int(rule_Len)>1:
                for r in rule_dict1[rule_Len]:
                        res_dict = {}
                        for idx in query_idx:
                            if train_edges[idx][1] == rel_idx:
                                line = train_edges[idx]
                                with open('../output/found_rules/'+ self.dataset_using \
                                                    +'_train_query_'+str(idx)+'.json', 'r') as f:
                                    v = json.load(f)
                                if rule_Len in v.keys():
                                    cur_rules = [w['rule'] for w in v[rule_Len]]
                                    # print(cur_rules)
                                    if r['rule'] in cur_rules:
                                        # print(line)
                                        masked_facts = np.delete(train_edges, \
                                                    [idx, self.get_inv_idx(len(train_edges)//2, idx)], 0)
                                        X = self.create_mapping_facts(masked_facts, line[3:], 1)
                                        x = np.unique(X[:,:4], axis=0)
                                        _, rule_walks = self.apply_single_rule(line[0], line[3:], r['rule'], int(rule_Len),
                                                                                            x, return_walk=1)
                                        rule_walks = self.obtain_sp_rule_walks(rule_walks, r['rule'], int(rule_Len), X)
                                        # print(rule_walks)
                                        y = rule_walks[rule_walks[:,-1]==line[2],:-1].tolist()
                                        cur_dict = dict(Counter([tuple(y1) for y1 in y]))
                                        res_dict = self.my_merge_dict(res_dict, cur_dict)

                        rule_sup[str(r['rule'])] = res_dict
        return rule_sup


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


    def find_common_nodes(self, ls1, ls2):
        return [list(set(ls1[i]).intersection(set(ls2[i]))) for i in range(len(ls1))]


    # def generate_input_dict_v2(self, rel_idx, ruleLen, train_edges):
    #     num_paths_max, num_path_sampling = 20, 5
    #     cur_num_paths = min(num_paths_max, self.num_paths_dict[rel_idx][ruleLen])

    #     input_dict_ls = []
    #     for (i, query) in enumerate(train_edges):
    #         if query[1] == rel_idx:
    #             cur_idx_inv = self.get_inv_idx(int(len(train_edges)/2), i)

    #             cur_file_path = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(i)+'.json'
    #             cur_file_path_inv = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(cur_idx_inv)+'.json'

    #             if os.path.exists(cur_file_path) or os.path.exists(cur_file_path_inv):
    #                 if os.path.exists(cur_file_path):
    #                     with open(cur_file_path) as f:
    #                         ent_walk_res = np.array(json.load(f))
    #                 else:
    #                     with open(cur_file_path_inv) as f:
    #                         ent_walk_res = np.array(json.load(f))[:,::-1]

    #                 if ent_walk_res.shape[1] -1 == ruleLen:
    #                     print(ent_walk_res)
    #                     masked_facts = np.delete(train_edges, [i, cur_idx_inv], 0)
    #                     query_int = query[3:]

    #                     ent_walk_res_ls = []
    #                     if ent_walk_res.shape[0]>cur_num_paths:
    #                         for m in range(num_path_sampling):
    #                             np.random.shuffle(ent_walk_res)
    #                             ent_walk_res_ls.append(ent_walk_res[:cur_num_paths,:])
    #                     else:
    #                         ent_walk_res_ls.append(ent_walk_res)

    #                     cur_input_dicts = []
    #                     for cur_walk_mat in ent_walk_res_ls:
    #                         cur_input_dict = {'rel_dict': {}, 'tmp_rel_dict': {}}
    #                         cur_mask = []
    #                         for l in range(cur_num_paths):
    #                             cur_input_dict['rel_dict'][l] = {}
    #                             cur_input_dict['tmp_rel_dict'][l] = {}

    #                             if l<cur_walk_mat.shape[0]:
    #                                 cur_mask.append(0)
    #                                 for k in range(ruleLen):
    #                                     rel, tmp_rel = self.obtain_edges_given_nodes(masked_facts, cur_walk_mat[l,k], cur_walk_mat[l,k+1], query_int)
    #                                     cur_input_dict['rel_dict'][l][k] = rel
    #                                     cur_input_dict['tmp_rel_dict'][l][k] = tmp_rel
    #                             else:
    #                                 cur_mask.append(-1000)
    #                                 for k in range(ruleLen):
    #                                     cur_input_dict['rel_dict'][l][k] = [0]
    #                                     cur_input_dict['tmp_rel_dict'][l][k] = [0]
    #                         cur_input_dict['mask'] = cur_mask
    #                         cur_input_dicts.append(cur_input_dict)

    #                     input_dict_ls.append(cur_input_dicts)

    #                     if len(input_dict_ls)>self.num_train_samples_max:
    #                         break

    #     return input_dict_ls


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
                    
            df_edges[i] = df_edges[i][0:0]
        
        return rule_walks


    def get_weights_savepath(self, rel_idx, ruleLen, train_idx):
        return self.weights_savepath + '_rel_'+ str(rel_idx) +'_len_'+ str(ruleLen) + '_idx_'+ str(train_idx) +'.json'

    def get_weights_savepath_v2(self, rel_idx):
        return self.weights_savepath + '_rel_'+ str(rel_idx) +'.json'


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


    def obtain_sp_rule_walks(self, ent_walks, rule, rule_Len, facts):
        rel_ls = rule[:rule_Len]
        pattern_ls = rule[rule_Len:]

        path_ls = []
        for i in range(rule_Len):
            y = facts[np.isin(facts[:, 0], ent_walks[:,i]) & (facts[:, 1] == rel_ls[i]) & \
                                np.isin(facts[:, 2], ent_walks[:,i+1]) & (facts[:, 3] == pattern_ls[i])]
            path_ls.append(y[:, [0,4,5,2]])

        # print(path_ls)

        z = ['entity_', 'ts_', 'te_']
        cur_ent_walk_res = self.get_walks(path_ls, z).to_numpy()
        path_ls = []

        sp_ls = []
        for i1 in range(1, rule_Len):
            for i2 in range(i1):
                # print(i1,i2)
                x = self.obtain_tmp_rel(cur_ent_walk_res[:,3*i1+1:3*i1+3], cur_ent_walk_res[:,3*i2+1:3*i2+3])
                sp_ls.append(x.reshape(-1,1))

        y = cur_ent_walk_res[:, [3*i1 for i1 in range(rule_Len)]]

        x = np.hstack([y, np.hstack(sp_ls), cur_ent_walk_res[:,-1:]])
        x = np.unique(x, axis=0)
        return x[:, rule_Len:]


    def obtain_tmp_rel(self, int1, int2, tol_gap=[0, 0]):
        if len(int2.shape) == 1:
            return ((int1[:, 1] <= int2[0] - tol_gap[0]) & (int1[:, 0] <= int2[0] - tol_gap[0]))*(-1) + ((int1[:, 0] >= int2[1] + tol_gap[1]) & (int1[:, 1] >= int2[1] + tol_gap[1]))*1
        else:
            return ((int1[:, 1] <= int2[:,0] - tol_gap[0]) & (int1[:, 0] <= int2[:,0] - tol_gap[0]))*(-1) + ((int1[:, 0] >= int2[:,1] + tol_gap[1]) & (int1[:, 1] >= int2[:,1] + tol_gap[1]))*1


    def collect_rules(self, idx_ls, rel_idx):
        if not isinstance(idx_ls, list):
            idx_ls = [idx_ls]

        rules_dict = []
        for idx in idx_ls:
            if self.overall_mode == 'general':
                path = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path = '../output/found_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'

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
        # mask_pre: [1, 1, 1, 0, 0, ...] decide which notations to preserve
        alpha_dict = {}
        for r in rules_dict:
            cur_rule = tuple([a*b for a, b in zip(r['rule'], mask_pre)])
            if cur_rule not in alpha_dict:
                alpha_dict[cur_rule] = []
            alpha_dict[cur_rule].append(r['alpha'])
        return alpha_dict


    def merge_dicts(self, dict_list, num_samples):
        merged_dict = defaultdict(list)
        for d in dict_list:
            merged_dict[tuple(d['rule'])].append(d['alpha'])
        merged_dict = dict(merged_dict)
        for rule in merged_dict:
            merged_dict[rule] = np.mean(merged_dict[rule] + [0] * (num_samples - len(merged_dict[rule])))
        return merged_dict


    def prepare_inputs(self, idx_ls, const_pattern_ls, rel_idx, TR_ls_len):
        '''
        If we process multiple samples at one time, the problem is that A can vote 10 times if it has 10 rules but B can only vote 5 times if it has 5 rules.
        We can process each sample separately but it is time-consuming.
        Instead, we merge the rule dict and calculate the average alpha.
        '''
        idx_ls = [idx_ls] if not isinstance(idx_ls, list) else idx_ls

        rules_dict, shallow_rules_dict = {}, {}
        for idx in idx_ls:
            if self.overall_mode == 'general':
                path = '../output/found_rules/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path = '../output/found_rules_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+'.json'
            # print(path)

            if not os.path.exists(path):
                # print(path + ' not found')
                cur_rules_dict = {}
            else:
                with open(path, 'r') as f:
                    cur_rules_dict = json.load(f)

            for rule_len in cur_rules_dict.keys():
                if rule_len not in rules_dict.keys():
                    rules_dict[rule_len] = []
                rules_dict[rule_len] += cur_rules_dict[rule_len]

                if rule_len not in shallow_rules_dict.keys():
                    shallow_rules_dict[rule_len] = []

                alpha_dict = self.merge_rules(cur_rules_dict[rule_len], [1] * (2*int(rule_len) + int(rule_len)*(int(rule_len)-1)/2))
                shallow_rules_dict[rule_len] += [{'rule': r, 'alpha': np.mean(alpha_dict[r])} for r in alpha_dict]
        
        for rule_len in rules_dict:
            rules_dict[rule_len] = self.merge_dicts(rules_dict[rule_len], len(idx_ls))
            rules_dict[rule_len] = [{'rule': list(key), 'alpha': value} for key, value in rules_dict[rule_len].items()]

            shallow_rules_dict[rule_len] = self.merge_dicts(shallow_rules_dict[rule_len], len(idx_ls))
            shallow_rules_dict[rule_len] = [{'rule': list(key), 'alpha': value} for key, value in shallow_rules_dict[rule_len].items()]

    
        input_dict = {}
        f_valid = 0
        shallow_rule_idx, shallow_rule_alpha = np.zeros((1, self.shallow_score_length)), np.zeros((1, self.shallow_score_length))
  
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
                    input_dict[i] = {'rel': np.array(rel_ls), 'TR': np.array(TR_ls), 'alpha': np.array(alpha_ls)}
                    f_valid = 1
            else:
                input_dict[i] = {'rel': np.zeros((1, i+1)), 'TR': np.zeros((1, TR_ls_len[i])), 'alpha': np.zeros((1, 1))}

            if rule_len in shallow_rules_dict.keys():
                for r in shallow_rules_dict[rule_len]:
                    cur_shallow_rule = r['rule']
                    if cur_shallow_rule not in self.shallow_rule_dict[rel_idx]:
                        continue
                    shallow_rule_idx[0, self.shallow_rule_dict[rel_idx].index(cur_shallow_rule)] = 1
                    shallow_rule_alpha[0, self.shallow_rule_dict[rel_idx].index(cur_shallow_rule)] += r['alpha']

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
                path = '../output/found_t_s/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json'
            elif self.overall_mode in ['few', 'biased', 'time_shifting']:
                path = '../output/found_t_s_'+ self.overall_mode +'/'+ self.dataset_using +'_train_query_'+str(idx)+ '.json'

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


    def variable_init_v3(self):
        prob_rel_ls = []
        prob_pattern_ls = []
        prob_ruleLen = tf.nn.softmax(tf.Variable(np.random.rand(1, self.num_ruleLen), dtype=tf.float32), axis=1)

        for i in range(1, self.num_ruleLen+1):
            prob_rel = tf.nn.softmax(tf.Variable(np.random.rand(self.num_rel, i), dtype=tf.float32), axis=0)
            prob_rel = tf.split(prob_rel, i, axis=1)
            prob_rel_ls.append(prob_rel)

            prob_pattern = tf.nn.softmax(tf.Variable(np.random.rand(self.num_pattern, i), dtype=tf.float32), axis=0)
            prob_pattern = tf.split(prob_pattern, i, axis=1)
            prob_pattern_ls.append(prob_pattern)

        return prob_rel_ls, prob_pattern_ls, prob_ruleLen


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