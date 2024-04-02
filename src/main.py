import sys
import json

from Models import *
from dataset_setting import *
from dataset_setting2 import *
from exploring import *
from training import *
from apply_rules import *
from create_all_rules_dict import *
from prediction import *
from other_funs import *




dataset_selection = 1 # select dataset
dataset_using = ['wiki', 'YAGO'][dataset_selection]
steps_to_do = ['find_rules', 'train_models', 'create_rule_dicts', 'predict', 'evaluate']




num_pattern = 3 # num of temporal relations: before, touching, after
const_pattern_ls = [-1, 0, 1] # notations for temporal relations
num_ruleLen = [3, 5][dataset_selection]  # max rule length

mode_selection = 0 # select mode: general, few training samples, biased data, time shifting
overall_mode = ['general', 'few', 'biased', 'time_shifting'][mode_selection] 

if dataset_using == 'wiki':
    dataset_name1 = 'WIKIDATA12k'
    num_rel = 48
    num_entites = 12554
elif dataset_using == 'YAGO':
    dataset_name1 = 'YAGO11k'
    num_rel = 20
    num_entites = 10623


para_ls_for_model = [num_rel, num_pattern, num_ruleLen, dataset_using, overall_mode]
targ_rel_ls = range(num_rel)
f_use_tfm = 0  # whether use temporal feature modeling





if not os.path.exists('../output'):
    os.mkdir('../output')
output_files = ['found_rules', 'found_t_s', 'train_weights_tfm', 'train_weights', 'learned_rules', 'explore_res', 'rank_dict']
for filename in output_files:
    if not os.path.exists('../output/' + filename):
        os.mkdir('../output/' + filename)



train_edges, valid_data, valid_data_inv, test_data, test_data_inv = do_normal_setting_for_dataset(dataset_name1, num_rel)
assiting_data = obtain_assiting_data(dataset_using, dataset_name1, train_edges, num_rel, num_entites)

# To accelarate, we randomly select 100 positive samples for each relation
pos_examples_idx, bg_train, bg_pred = split_dataset(dataset_using, dataset_name1, train_edges, num_rel, num_sample_per_rel=100)


print('Num of pos examples used:', len(pos_examples_idx))


dist_pars = []
if f_use_tfm:
    p_rec, p_order, mu_pair, sigma_pair, lambda_pair, mu_dur, sigma_dur = obtain_distribution_parameters(train_edges, num_rel)
    dist_pars += [p_rec, p_order, mu_pair, sigma_pair, lambda_pair]



if 'find_rules' in steps_to_do:
    print('Start finding rules.')
    do_my_find_rules(targ_rel_ls, bg_train, para_ls_for_model, pos_examples_idx = pos_examples_idx)

    print('Finding rules finished.')





if 'train_models' in steps_to_do:
    print('Start training model.')
    do_my_train_TRL(para_ls_for_model, bg_train, const_pattern_ls,
                            pos_examples_idx = pos_examples_idx, targ_rel_ls = targ_rel_ls)

    if f_use_tfm:
        do_my_train_tfm(para_ls_for_model, targ_rel_ls, train_edges, dist_pars)

    print('Training model finished.')



if 'create_rule_dicts' in steps_to_do:
    print('Start creating rule dicts.')
    do_calculate_rule_scores(targ_rel_ls, para_ls_for_model, 
                                bg_train, const_pattern_ls, 
                                pos_examples_idx = pos_examples_idx)
    print('Creating rule dicts finished.')







if 'predict' in steps_to_do:
    print('Start prediction:')

    do_my_predict(targ_rel_ls, para_ls_for_model, bg_pred, valid_data, valid_data_inv, 
                    const_pattern_ls, assiting_data, dist_pars, train_edges)

    if 'evaluate' in steps_to_do:
        print('Validation set:')
        res_dict = do_evaluate(targ_rel_ls, dataset_using)

        with open('../output/'+ dataset_using +'_valid_results.json', 'w') as f:
            json.dump(res_dict, f)


    do_my_predict(targ_rel_ls, para_ls_for_model, bg_pred, test_data, test_data_inv, 
                    const_pattern_ls, assiting_data, dist_pars, train_edges)

    if 'evaluate' in steps_to_do:
        print('Test set:')
        res_dict = do_evaluate(targ_rel_ls, dataset_using)

        with open('../output/'+ dataset_using +'_test_results.json', 'w') as f:
            json.dump(res_dict, f)

    print('Prediction finished.')