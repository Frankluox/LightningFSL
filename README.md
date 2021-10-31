# Few-Shot Learning in Pytorch-Lightning
A friendly codebase for Few-Shot Learning based on pytorch-lightning.


## General guide:
To get started, the followings need to be done:

    * Choose a configuration file in `config(e.g.`set_config_PN.py` for PN model).
    * modify the first line in run.sh (e.g., `python config/set_config_PN.py`).
    * To begin the running: 
    
&nbsp; &nbsp; &nbsp; &nbsp; `$ bash run.sh`






## Download available datasets (will be gradually updated)

[miniImageNet](https://1drv.ms/u/s!AkYSH77Z8H6qa872NXTDnt-6bwY?e=XcKJgH)

## Few-shot classification Results (will be gradually updated)
Implemented results on few-shot learning datasets with ResNet-12 backbone. The average results of 2,000 randomly sampled episodes repeated 5 times for 1/5-shot evaluation with 95% confidence interval are reported. 

#### miniImageNet Dataset

|Models|5-way 1-shot|5-way 5-shot|
|:----:|:----:|:----:|
|[Protypical Networks](https://arxiv.org/abs/1703.05175)|61.19+-0.40 |  76.50+-0.45| 
|[Cosine Classifier](https://arxiv.org/abs/1804.09458)|63.89+-0.44|80.94+-0.05|
|[Meta-Baseline](https://arxiv.org/abs/2003.04390)|62.65+-0.65|79.10+-0.29|
|[S2M2](https://arxiv.org/abs/1907.12087)|58.85+-0.20|81.83+-0.15|
|S2M2+Logistic_Regression|62.36+-0.42|82.01+-0.24|
|[MoCo-v2](https://arxiv.org/abs/1911.05722)(unsupervised)|52.03+-0.33|72.94+-0.29|
|[Exemplar-v2](https://arxiv.org/abs/2006.06606)|59.02+-0.24|77.23+-0.16|
|[PN+CL](https://arxiv.org/abs/2107.09242)|63.44+-0.44|79.42+-0.06|
|[COSOC](https://arxiv.org/abs/2107.07746)|69.28+0.49|85.16+-0.42|

## Advanced:

Write your own model in `modules` and config file in `config`, run configuration file first to generate `config.yaml`, and execute `run.py` file to run the program.

