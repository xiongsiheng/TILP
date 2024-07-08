import sys
import json

from Models import *
from dataset_setting import *
from training import *
from apply_rules import *
from create_all_rules_dict import *
from prediction import *
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--find_rules', action="store_true")
parser.add_argument('--train_model', action="store_true")
parser.add_argument('--rule_summary', action="store_true")
parser.add_argument('--predict', action="store_true")
parser.add_argument('--rule_len', type=int)
parser.add_argument('--mode', default='general', type=str)
parser.add_argument('--use_tfm', action="store_true")


args = parser.parse_args()


dataset_using = args.dataset
num_ruleLen = args.rule_len

# Possible modes: ['general', 'few', 'biased', 'time_shifting'];
# Meaning: general, few training samples, biased data, time shifting (Todo: fix issues for difficult modes)
overall_mode = args.mode

steps_to_do = []
if args.find_rules:
    steps_to_do.append('find rules')
if args.train_model:
    steps_to_do.append('train model')
if args.rule_summary:
    steps_to_do.append('rule summary')
if args.predict:
    steps_to_do.append('predict')

f_use_tfm = args.use_tfm  # whether use temporal feature modeling (Todo: fix issues)




num_pattern = 3 # num of temporal relations: before, touching, after
const_pattern_ls = [-1, 0, 1] # notations for temporal relations

if dataset_using == 'wiki':
    dataset_name1 = 'WIKIDATA12k'
    num_rel = 48
    num_entites = 12554
elif dataset_using == 'YAGO':
    dataset_name1 = 'YAGO11k'
    num_rel = 20
    num_entites = 10623


model_paras = [num_rel, num_pattern, num_ruleLen, {}, dataset_using, overall_mode]
targ_rel_ls = range(num_rel)


# create output folders
if not os.path.exists('../output'):
    os.mkdir('../output')
output_files = ['found_paths', 'found_time_gaps', 'train_weights_tfm', 'train_weights', 'learned_rules', 'explore_res', 'rank_dict']
for filename in output_files:
    if not os.path.exists('../output/' + filename):
        os.mkdir('../output/' + filename)



train_edges, valid_data, valid_data_inv, test_data, test_data_inv = do_normal_setting_for_dataset(dataset_name1, num_rel)
assiting_data = obtain_assiting_data(dataset_using, dataset_name1, train_edges, num_rel, num_entites)

# To accelarate, we randomly select a fixed number of positive samples for each relation. 
# To use all positive samples, set num_sample_per_rel = -1
pos_examples_idx, bg_train, bg_pred = split_dataset(dataset_using, dataset_name1, train_edges, num_rel, num_sample_per_rel=-1)


# print config information
print('######################### TILP Config #########################')
print('Dataset: '+ dataset_using)
print('Mode: '+ overall_mode)
print('Steps to do: '+ str(steps_to_do))
print('Num of pos examples used: ' + str(len(pos_examples_idx)))
print('Max rule length: ' + str(num_ruleLen))
print('Use temporal feature modeling: ' + str(f_use_tfm))
print('################################################################')



dist_pars = []
if f_use_tfm:
    p_rec, p_order, mu_pair, sigma_pair, lambda_pair, mu_dur, sigma_dur = obtain_distribution_parameters(train_edges, num_rel)
    dist_pars += [p_rec, p_order, mu_pair, sigma_pair, lambda_pair]



if 'find rules' in steps_to_do:
    do_my_find_rules(targ_rel_ls, bg_train, model_paras, mode='path_search', pos_examples_idx = pos_examples_idx, num_processes=24)
    do_rule_summary(targ_rel_ls, model_paras, num_rel, bg_train, const_pattern_ls, num_processes=24)
    do_my_find_rules(targ_rel_ls, bg_train, model_paras, mode='alpha_calculation', pos_examples_idx = pos_examples_idx, num_processes=24)



if 'train model' in steps_to_do:
    # To accelarate, in each epoch, we randomly select a fixed number of positive samples for each relation. 
    # To use all positive samples, set num_sample_per_rel = -1.
    do_my_train_TRL(model_paras, dataset_using, bg_train, const_pattern_ls, targ_rel_ls=targ_rel_ls, num_epoch=100, num_sample_per_rel=-1)

    if f_use_tfm:
        do_my_train_tfm(model_paras, targ_rel_ls, train_edges, dist_pars)



if 'rule summary' in steps_to_do:
    do_calculate_rule_scores(targ_rel_ls, model_paras, num_rel, bg_train, const_pattern_ls, num_processes=24)



if 'predict' in steps_to_do:
    do_my_predict(targ_rel_ls, model_paras, dataset_using, bg_pred, valid_data, valid_data_inv, 
                    const_pattern_ls, assiting_data, dist_pars, train_edges, num_processes=24)

    print('Validation set:')
    res_dict = do_evaluate(targ_rel_ls, dataset_using)

    with open('../output/'+ dataset_using +'_valid_results.json', 'w') as f:
        json.dump(res_dict, f)


    do_my_predict(targ_rel_ls, model_paras, dataset_using, bg_pred, test_data, test_data_inv, 
                    const_pattern_ls, assiting_data, dist_pars, train_edges, num_processes=24)

    print('Test set:')
    res_dict = do_evaluate(targ_rel_ls, dataset_using)

    with open('../output/'+ dataset_using +'_test_results.json', 'w') as f:
        json.dump(res_dict, f)