"""
Make a model InD vs OOD comparisong
"""
import hydra

import pandas as pd
import numpy as np
import random
from ensemble_testbed.predictions import EnsembleModel
import yaml
import os
from pathlib import Path

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
@hydra.main(config_path="../configs/experiments/model_comparison",
            config_name="imagenet")
def main(args):

    #%%
    # get a random seed
    if args.seed is None:
        seed = np.random.randint(0, 100000)
    random.seed(seed)

    #%% read metrics_file
    ind_metrics_file = output_dir / "model_performance/{}.csv".format(
        args.dataset)
    ood_metrics_file = output_dir / "model_performance/{}.csv".format(
        args.ood_dataset)

    ind_models = pd.read_csv(ind_metrics_file)
    #ind_models['dataset'] = args.dataset
    ood_models = pd.read_csv(ood_metrics_file)
    #ood_models['dataset'] = args.ood_dataset

    #%% rename some
    merge_factors = ['models', 'train_set']
    for col in ood_models.columns:
        if col in merge_factors:
            continue
        ood_models = ood_models.rename(columns={col: 'shift_{}'.format(col)})

    df_results = ind_models.merge(ood_models, on=merge_factors, how='inner')
    df_results.rename(columns={'shift_test_set': 'shift_set'}, inplace=True)
    results_filename = output_dir / "results_model_comparison.csv"

    # import pdb; pdb.set_trace()
    #if os.path.exists(results_filename):
    #    df_results = pd.concat([pd.read_csv(results_filename, index_col=False), df_results], axis=0)
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

    bias, var, perf = ens.get_avg_nll(), ens.get_nll_div(), ens.get_nll()
    print(
        'Built ensemble {}: \n nparams {} player_score {} with bias {:.3f}  var {:.3f}  perf {:.3f}'
        .format(name, num_params, all_player_scores, bias, var, perf),
        flush=True)
    del ens
    return [
        name, bias, var, perf, num_params, all_player_scores, all_model_names
    ]


#%%
if __name__ == "__main__":
    main()
