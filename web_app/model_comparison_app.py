"""Main application logic.
Edited from https://github.com/millerjohnp/linearfits_app/app.py
"""
import ast
import base64

import numpy as np
import pandas as pd
import streamlit as st

import utils

st.write("""
## [Ensemble Testbed](https://github.com/ekellbuch/ensemble_testbed)
[Kelly Buchanan](https://ekbuchanan.com/)

Evaluate the performance of models from a variety of open source models.

"""
)


from pathlib import Path
import os

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

output_dir = BASE_DIR / "results"


@st.cache_data
def load_data():
    df = pd.read_csv(output_dir / "results_model_comparison.csv",
        converters={
            "acc": ast.literal_eval,
            "shift_acc": ast.literal_eval,
        })
    # df["model_type"] = df.apply(utils.get_model_type, axis=1)
    return df

df = load_data()

universe = st.sidebar.selectbox(
    "Dataset universe", ["ImageNet"], index=0)

metrics = ["acc", "nll", "brier", "qunc"]

if universe == "ImageNet":
    shift_type = st.sidebar.selectbox(
        "Distribution shift type", [
            "Dataset reproduction",
            "Benchmark shift",
            "Synthetic perturbations",
        ],
        index=0
    )
    train_sets = ["imagenet"]
    test_sets = ["imagenet"]
    shift_sets =["imagenetv2mf"]

train_set = st.sidebar.selectbox(
    "Train dataset:", train_sets, index=0)
test_set = st.sidebar.selectbox(
    "Test dataset (x-axis):", test_sets, index=0)
shift_set = st.sidebar.selectbox(
    "Shift dataset (y-axis):", shift_sets, index=0)

selected_df = df[
    (df.train_set == train_set)
    & (df.test_set == test_set)
    & (df.shift_set == shift_set)
]


scaling = st.sidebar.selectbox(
    "Axis scaling:", ["probit", "logit", "linear"], index=0)

metric = st.sidebar.selectbox(
    "Metric: ", metrics, index=0)

if st.sidebar.checkbox(f"Show only a subset of models?", value=False):
    model_types = list(selected_df.model_type.unique())
    types_to_show = set(st.sidebar.multiselect(f"Models to show", options=model_types))
    if len(types_to_show):
        selected_df = selected_df[selected_df.model_type.isin(types_to_show)]

st.plotly_chart(utils.plot(selected_df, scaling=scaling, metric=metric))

"To visualize only a subset of model types (e.g. just Linear Models), check the box to show only a subset of models in the left sidebar."

"Click the link below to download the raw data for this plot as a csv"
# Encode data for download
df_to_download = selected_df.to_csv(index=False)
b64 = base64.b64encode(df_to_download.encode()).decode()

linkname = f"train:{train_set}_test:{test_set}.csv"
link = f'<a href="data:file/txt;base64,{b64}" download="{linkname}"> Download data as csv</a>'
st.markdown(link, unsafe_allow_html=True)

st.write("""
### Citation
```
@inproceedings{miller2021accuracy,
    title={Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization},
    author={Miller, John and Taori, Rohan and Raghunathan, Aditi and Sagawa, Shiori and Koh, Pang Wei and Shankar, Vaishaal and Liang, Percy and Carmon, Yair and Schmidt, Ludwig},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2021},
    note={\\url{https://arxiv.org/abs/2007.00644}},
}
```
""")