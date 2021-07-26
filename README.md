# Few-Shot Learning in Pytorch-Lightning
A friendly codebase for Few-Shot Learning based on pytorch-lightning.


## Usage:
Choose a configuration file in `config` (e.g., `set_config_PN.py` for PN model), then modify the first line in run.sh, e.g.,

`python config/set_config_PN.py`

To begin the running:

`$ bash run.sh`

## Advanced:

Write your own model in `modules` and config file in `config`, run configuration file first to generate `config.yaml`, and execute `run.py` file to run the program.

## Few-shot classification Results
Implemented results on few-shot learning datasets with ResNet-12 backbone. The average results of 2,000 randomly sampled episodes repeated 5 times for 1/5-shot evaluation with 95% confidence interval are reported.

#### miniImageNet Dataset

|Models|5-way 1-shot|5-way 5-shot|
|:----:|:----:|:----:|
|[Protypical Networks](https://arxiv.org/abs/1703.05175)|61.19+-0.40 |  76.50+-0.45| 
|[Cosine Classifier](https://arxiv.org/abs/1804.09458)|63.89+-0.44|80.94+-0.05|
