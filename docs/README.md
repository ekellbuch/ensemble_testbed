Follow the steps below to use your own model logits and labels and calculate ensemble metrics.

## Installation:
Instructions using conda:

Create a new environment:
```
conda create -n ensemble_testbed python=3.7
```

Now move into the root directory of this repo:
```
cd /path/to/this/repo
```

Activate your new environment, install dependencies and python package: 
```
conda activate ensemble_testbed
conda install pip 
pip install -r requirements.txt
pip install -e ./src
```


## Experiments

Experiments to add your own model logits:
### Single models:
1. [x] Create .yaml file with logit and label paths:
```
# See configs/datasets/imagenet/imagnet.yaml for an example
```
1. [x] Make file with single model metrics:
``` 
python scripts/calculate_model_performance.py "--config-name=imagenet"
```
2. [x] Combine InD and OOD files:
```
python scripts/metrics_model_comparison.py
```
3. [x] Visualize model in streamlit
```
streamlit run web_app/model_comparison_app.py
```

### Ensembles:

1. [x] Calculate ensemble metrics:
```
python scripts/metrics_het_ensemble_parallel.py --dataset=imagenet
```

2. [x] Visualize model in streamlit
```
streamlit run web_app/ensemble_gains_app.py
```

## Next steps:
1.[ ] Include additional datasets.
2.[ ] Include other forms of ensembling.