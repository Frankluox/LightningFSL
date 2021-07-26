# Few-Shot Learning in Pytorch-Lightning
A friendly codebase for Few-Shot Learning based on pytorch-lightning.


## Usage:

Simple usage for running Prototypical Network (temporarily only supported):

bash run.sh

## Advanced:

Write your own model and set_config file, then run set_config file first to generate config.yaml, then execute run.py file to run the program.

## Few-shot classification Results
Implemented results on few-shot learning datasets with ResNet-12 backbone. The average results of 2,000 randomly sampled episodes repeated 5 times for 1/5-shot evaluation with 95% confidence interval are reported.

#### miniImageNet Dataset

|Models|5-way 1-shot|5-way 5-shot|
|:----:|:----:|:----:|
|[Protypical Networks](https://arxiv.org/abs/1703.05175)|61.19\$\pm$\0.40 |  76.50\$\pm$\0.45| 
