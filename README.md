## Uncertainty-Aware Search Framework for Multi-Objective Bayesian Optimization with Constraints


This repository contains the python implementation for USeMOC from the ICML 2020 Workshop on Automated Machine Learning (AutoML) "[Uncertainty-Aware Search Framework for Multi-Objective Bayesian Optimization with Constraints](https://arxiv.org/abs/2008.07029)".

The implementation handles automatically the batch version of the algorithm by setting the variable "batch_size" to a number higher than 1. 


### Requirements
The code is implemented in Python and requires the following packages:

1. [platypus](https://platypus.readthedocs.io/en/latest/getting-started.html#installing-platypus)

2. [sklearn.gaussian_process](https://scikit-learn.org/stable/modules/gaussian_process.html)

3. [pygmo](https://esa.github.io/pygmo2/install.html) 

### Citation

If you use this code please cite our paper:

```bibtex

@article{belakaria2020uncertainty,
  title={Uncertainty aware Search Framework for Multi-Objective Bayesian Optimization with Constraints},
  author={Belakaria, Syrine and Deshwal, Aryan and Doppa, Janardhan Rao},
  journal={Workshop on Automated Machine Learning (AutoML), ICML},
  year={2020}
}

````
