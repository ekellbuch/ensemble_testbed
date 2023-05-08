"""
Ensemble bias var decomposition
"""
import ast
import base64

import numpy as np
import pandas as pd
import streamlit as st

import utils_ensemble

st.write("""
## [Ensemble Testbed](https://github.com/ekellbuch/ensemble_testbed)
[Kelly Buchanan](https://ekbuchanan.com/)

Evaluate the performance of ensembles from a variety of open source models.
"""
)


from pathlib import Path
import os

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

output_dir = BASE_DIR / "results"


@st.cache_data
def load_data():
    df = pd.read_csv(output_dir / "bias_var_msize/ens_binned_values_scored_parallel.csv",
        converters={
            #"hyperparameters": ast.literal_eval,
            "bias": ast.literal_eval,
            "var": ast.literal_eval,
            "perf": ast.literal_eval,
        })
    # df["model_type"] = df.apply(utils.get_model_type, axis=1)
    return df

df = load_data()

universe = st.sidebar.selectbox(
    "Dataset universe", ["ImageNet"], index=0)

# where to fix ens_size?
metrics = ['Ensemble Accuracy',
           'Total Variance',
           'Avg. NLL', 'Ensemble Diversity (NLL)', 'Ensemble NLL',
           'Num. Params']

if universe == "ImageNet":
    dataset = ["imagenet", "imagenetv2mf"]

train_set = st.sidebar.selectbox(
    "Dataset:", dataset, index=0)

selected_df = df[
    (df.dataset == train_set)
]

scaling = st.sidebar.selectbox(
    "Axis scaling:", ["linear"], index=0)

metric_x = st.sidebar.selectbox(
    "(x-axis): ", metrics, index=1)

metric_y = st.sidebar.selectbox(
    "(y-axis): ", metrics, index=0)


add_linear_fit = st.sidebar.checkbox(f"Add x=y line?", value=False)

st.plotly_chart(utils_ensemble.plot(selected_df,
                                    scaling=scaling,
                                    metric_x=metric_x,
                                    metric_y=metric_y,
                                    add_linear_fit=add_linear_fit))


