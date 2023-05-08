"""
Make ensembles for imagenet, and calculate bias/variance for each ensemble size.
"""
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import combinations
import random
from tqdm import tqdm
from ensemble_testbed.predictions import EnsembleModel
import multiprocessing
from functools import partial
import fire
import yaml
import os

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
output_dir = BASE_DIR / "results"


# read size of models trained on imagenet
def read_model_size_file(dataset):
    if 'imagenet' in dataset:
        dataset = 'imagenet'
    model_size_file = BASE_DIR / "configs/datasets/{}/model_num_params.yaml".format(
        dataset)
    with open(model_size_file) as f:
        my_dict = yaml.safe_load(f)
    return my_dict


def get_player_score(perf):
    player_perf_bins = np.linspace(0, 1, 10)
    return np.digitize(perf, player_perf_bins)


#%%
def main(ensemble_size=4,
         max_mpclass=11,
         max_enspclass=11,
         serial_run=False,
         seed=None,
         dataset='imagenet',
         out_name="bias_var_msize/ens_binned_values_scored_parallel"):

    # average models by binning them in terms of performance
    for binning in [3, 6, 8]:
        run_model(binning=binning,
                  ensemble_size=ensemble_size,
                  max_mpclass=max_mpclass,
                  max_enspclass=max_enspclass,
                  serial_run=serial_run,
                  seed=seed,
                  dataset=dataset,
                  out_name=out_name)


def run_model(binning=6,
              ensemble_size=4,
              max_mpclass=11,
              max_enspclass=11,
              serial_run=False,
              seed=None,
              dataset='imagenet',
              out_name="bias_var_msize/ens_binned_values_scored_parallel"):

    #%%
    # get a random seed
    if seed is None:
        seed = np.random.randint(0, 100000)
    random.seed(seed)
    #%% read metrics_file
    metrics_file = BASE_DIR / "results/model_performance/{}.csv".format(
        dataset)

    ind_models = pd.read_csv(metrics_file)

    #%% Filter out models in terms of performance
    err_values = 1 - ind_models['acc'].values

    # To populate plot, we need to average model in different bins
    if binning == 3:
        err_bins = np.linspace(0, 1, 4)
    elif binning == 6:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.asarray([0.3, 0.8, 1])
    elif binning == 8:
        # for het_ensemble_2 now use smaller bins
        err_bins = np.asarray([0.18, 0.20, 0.25, 0.35, 0.6, 1])

    #%%
    model_cl_assignment = np.digitize(err_values, err_bins)
    model_classes, num_models_p_class = np.unique(model_cl_assignment,
                                                  return_counts=True)

    print('Bins ', err_bins.round(2), num_models_p_class, flush=True)

    all_biasvar = []
    bin_id_list = []
    results = []
    model_num_params = read_model_size_file(dataset)['models']
    #%%
    for model_class_id in model_classes:
        #%% Find all models in group
        models_in_cls = np.argwhere(
            model_cl_assignment == model_class_id).flatten()
        if len(models_in_cls) <= ensemble_size:
            print('Skipping bin {}'.format(err_bins[model_class_id]),
                  flush=True)

        #%% control number of models in each bin:
        if len(models_in_cls) > max_mpclass:
            models_in_cls = random.sample(list(models_in_cls), max_mpclass)
        #%%
        ensemble_groups = set()
        num_ensembles = min(
            max_enspclass, len(list(combinations(models_in_cls,
                                                 ensemble_size))))
        while len(ensemble_groups) < num_ensembles:
            ensemble_groups.add(
                tuple(sorted(random.sample(list(models_in_cls),
                                           ensemble_size))))
        try:
            err_bins[model_class_id]
        except:
            import pdb
            pdb.set_trace()
        print('\nBuilding {} ensembles in bin {}\n'.format(
            len(ensemble_groups), err_bins[model_class_id]),
              flush=True)
        # run in parallel
        if serial_run:
            #"""
            for egroup_idx, ensemble_group in enumerate(tqdm(ensemble_groups)):
                tmp_outs = get_ensemble_metrics(ind_models, model_num_params,
                                                ensemble_group)
                all_biasvar.append(tmp_outs)
                bin_id_list.append([model_class_id])
            #"""
        else:
            get_ens = partial(get_ensemble_metrics, ind_models,
                              model_num_params)
            pool = multiprocessing.Pool(20)
            for result in tqdm(pool.imap_unordered(get_ens,
                                                   ensemble_groups,
                                                   chunksize=2),
                               total=len(ensemble_groups)):
                results.append([model_class_id] + result)

    #%%
    if serial_run:
        values = [bin_id_list, all_biasvar]
    else:
        values = results

    df_results = pd.DataFrame(values,
                              columns=[
                                  'bin_id', 'name', 'Avg. NLL',
                                  'Ensemble Diversity (NLL)', 'Ensemble NLL',
                                  'Ensemble Accuracy', 'Total Variance',
                                  'Num. Params', 'ensemble_hyperparameters'
                              ])

    df_results['ens_size'] = ensemble_size
    df_results['type'] = 'het'
    df_results['binning'] = binning
    df_results['seed'] = seed
    df_results['dataset'] = dataset

    results_filename = output_dir / (out_name + ".csv")
    print('The end, now storing data in {}'.format(results_filename),
          flush=True)
    os.makedirs(results_filename.parent, exist_ok=True)
    if os.path.exists(results_filename):
        df_results = pd.concat(
            [pd.read_csv(results_filename, index_col=False), df_results],
            axis=0)
    df_results.to_csv(results_filename, index=False)
    return


#%%
result_list = []


def log_result(result):
    result_list.append(result)


def get_ensemble_metrics(ind_models, model_num_params, ensemble_group):

    #model_num_params = read_model_size_file()['models']

    e_models = []
    for m_idx, m_ind_idx in enumerate(ensemble_group):
        model_ = ind_models.iloc[m_ind_idx]
        e_models.append(model_)
    # Ensemble name
    name = "--".join([m["models"] for m in e_models])
    # Make an ensemble:
    num_params = 0
    ens = EnsembleModel(name, "ind")
    player_scores = 0
    all_player_scores = []
    all_model_names = []

    for i, m in enumerate(e_models):
        e_model = m["models"]
        model_name = e_model + "_" + str(i)
        ens.register(
            filename=m["filepaths"],
            modelname=model_name,
            labelpath=m['labelpaths'],
        )
        # search for size of model
        acc = ens.models[model_name].get_accuracy()

        player_score = get_player_score(acc)
        player_scores += player_score
        num_params += model_num_params[m["models"]]['num_params']
        all_player_scores.append(player_score)
        all_model_names.append(e_model)

    bias, var, perf, acc, var2 = ens.get_avg_nll(), ens.get_nll_div(), \
        ens.get_nll(), ens.get_accuracy(), ens.get_variance()
    print(
        'Built ensemble {}: \n nparams {} player_score {} with bias {:.3f}  var {:.3f}  perf {:.3f}'
        .format(name, num_params, all_player_scores, bias, var, perf),
        flush=True)
    del ens
    additional_info = {
        'model_scores': all_player_scores,
        'model_names': all_model_names
    }
    return [name, bias, var, perf, acc, var2, num_params, additional_info]


#%%
if __name__ == "__main__":
    fire.Fire(main)
