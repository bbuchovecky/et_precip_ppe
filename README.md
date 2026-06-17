# Code for: "Reduced evapotranspiration and associated warming increase moisture convergence but decrease precipitation over land"

Author: Ben Buchovecky

Update the paths to model output in `src/et_precip_ppe/paths.py` then run the following commands to generate all figures in the manuscript:
```
conda env create -f environment.yml
conda activate et-precip-ppe
python scripts/make_all_figures_main.py
```

[![DOI](https://zenodo.org/badge/1117016829.svg)](https://doi.org/10.5281/zenodo.17945231)
