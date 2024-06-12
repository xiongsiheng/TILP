import json
from joblib import Parallel, delayed
from Models import *





def do_my_train_TRL(model_paras, dataset_using, train_edges, const_pattern_ls, 
                    targ_rel_ls = None, num_epoch=50, num_sample_per_rel=-1):
    my_model = Trainer(*model_paras)
    loss = my_model.TRL_model_training_v2(targ_rel_ls, num_epoch, train_edges, const_pattern_ls, num_sample_per_rel)
    loss_dict = {}
    loss_dict['loss_hist'] = [l.tolist() for l in loss]
    with open('../output/' + dataset_using +'_loss_dict.json', 'w') as f:
        json.dump(loss_dict, f)



def do_my_train_tfm(model_paras, rel_ls, train_edges, dist_pars, num_epoch = 100):
    my_model = Trainer(*model_paras)
    my_model.train_tfm_v2(rel_ls, num_epoch, train_edges, dist_pars)


def do_my_train_tfm_Wc(model_paras, rel_ls, train_edges, dist_pars, num_epoch = 100):
    my_model = Trainer(*model_paras)
    my_model.train_tfm_Wc_v2(rel_ls, num_epoch, train_edges, dist_pars)