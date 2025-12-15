# Code for: "Reduced evapotranspiration and associated warming increase moisture convergence but decrease precipitation over land"

Author: Ben Buchovecky

Run the following commands to generate all figures in the manuscript:
```
conda env create -f environment.yml
conda activate et_precip_ppe
python scripts/make_all_figures_main.py
python scripts/make_all_figures_supp.py
```