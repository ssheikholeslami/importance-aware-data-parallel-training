﻿
This repository contains the supplementary material for "The Impact of Importance-aware Dataset Partitioning on Data-parallel Training of Deep Neural Networks" published in [DAIS 2023](https://link.springer.com/chapter/10.1007/978-3-031-35260-7_5). Our author's version of the paper can be found [here](https://github.com/ssheikholeslami/importance-aware-data-parallel-training/blob/master/importance_aware_dpt_authors_version.pdf).

# Pre-requisites
### GPUs
As this system deals with data-parallel training of deep neural networks, you will need multiple CUDA-capable GPUs to run the code. The experiments in the paper were done using 4 Nvidia GPUs on a single machine running Ubuntu 18.04 LTS.
### Python Version
To run the code without any issues, we recommend that you create a Python virtual environment with **Python 3.8**, e.g., with `python3.8 -m venv env-name`, activate the environment using `source env-name/bin/activate` and then clone the repo and use `pip install -r requirements.txt` to install the specific versions of the required libraries. 

# Running the Experiments

- if you are interested in reproducing the box plots of Figure 5, you can use the `plots.ipynb` notebook. We have included CSV files of all our experiments in the `PaperCSVs` folder.

- if you want to re-run the exact same experiments in the paper yourself, please remember that you will need a worker with 4 Nvidia GPUs (please refer to Section 5.1 of the paper for our experimental setup) and it will take around 2 weeks in total to finish all the experiments. To run the experiments, simply execute the `run-experiments.sh` script, e.g., by `nohup ./run-experiments.sh &> logs.txt`.
- if you are interested in running only a specific set of experiments, you can comment out or modify the `run-experiments.sh` script.
- if you have a [Weights & Biases (W&B)](https://wandb.ai/) account, you can modify the code in `dpt.py` by searching for and uncommenting the lines that have `wandb` to manage the experiments using W&B.

# Cite
To cite this paper/software, you can use:
```
@inproceedings{sheikholeslami2023impact,
  title={The Impact of Importance-Aware Dataset Partitioning on Data-Parallel Training of Deep Neural Networks},
  author={Sheikholeslami, Sina and Payberah, Amir H and Wang, Tianze and Dowling, Jim and Vlassov, Vladimir},
  booktitle={IFIP International Conference on Distributed Applications and Interoperable Systems},
  pages={74--89},
  year={2023},
  organization={Springer},
  doi={10.1007/978-3-031-35260-7_5},
  url={https://doi.org/10.1007/978-3-031-35260-7_5}
}
```

# Acknowledgments
This work has been supported by the ExtremeEarth project funded by European Union’s Horizon 2020 Research and Innovation Programme under Grant Agreement No. 825258. The computations for some of the experiments were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) and the Swedish National Infrastructure for Computing (SNIC) at C3SE partially funded by the Swedish Research Council through grant agreement no. 2022-06725 and no. 2018-05973.
