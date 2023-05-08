"""
Calculate the model performance for a variety of models.
"""

import hydra
from ensemble_testbed.predictions import Model
import os
import pandas as pd
from pathlib import Path

here = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def get_arrays_toplot(models):
    """
  Takes as argument a dictionary of models:
  keys giving model names, values are dictionaries with paths to individual entries.
  :param models: names of individual models.
  """
    all_metrics = []
    for modelname, model in models.items():

        # for each model type there may be multiple entries:
        for i, (m, l, h) in enumerate(
                zip(model.filepaths, model.labelpaths, model.hyperparameters)):
            model = Model(m, "ind")
            model.register(
                filename=m,
                inputtype=None,
                labelpath=l,
                logits=True,
            )

            acc, nll, brier, qunc \
              = model.get_accuracy(), model.get_nll(), model.get_brier(), model.get_qunc()
            print("{}: Acc: {:.3f}, NLL: {:.3f}, Brier: {:.3f} Qunc:{:.3f}".
                  format(modelname, acc, nll, brier, qunc))
            all_metrics.append([modelname, m, l, acc, nll, brier, qunc, h])

    df = pd.DataFrame(all_metrics,
                      columns=[
                          "models", "filepaths", "labelpaths", "acc", "nll",
                          "brier", "qunc", "hyperparameters"
                      ])
    return df


@hydra.main(config_path="../configs/datasets/imagenet", config_name="imagenet")
def main(args):
    results_dir = Path(here) / "results/model_performance/{}.csv".format(
        args.title)
    os.makedirs(os.path.dirname(results_dir), exist_ok=True)

    print('\n dataset {} results_dir: {}\n'.format(args.title, results_dir))

    # Get performance metrics for each ensemble:
    df = get_arrays_toplot(args.models)

    # additional info:
    df['train_set'] = args.train_set
    df['test_set'] = args.test_set
    df['model_family'] = args.model_family

    # Dump to csv
    df.to_csv(str(results_dir))

    print('Stored performance of {} in {}'.format(args.title, results_dir))

    return


if __name__ == "__main__":
    main()
