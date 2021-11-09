# Few-Shot Learning in Pytorch-Lightning
A codebase for Few-Shot Learning (FSL) using the framework of [pytorch-lightning](https://www.pytorchlightning.ai/). A number of implementations of FSL algorithms are provided, including two official ones

[Boosting Few-Shot Classification with View-Learnable Contrastive Learning](https://arxiv.org/abs/2107.09242) (ICME 2021)

and

[Rectifying the Shortcut Learning of Background for Few-Shot Learning](https://arxiv.org/abs/2107.07746) (NeurIPS 2021)

Currently, we are sorry for potential drawbacks or bugs in our codes. We will make the codebase more robust in a few days.

## Contents
1. [Advantages](#Advantages)
2. [Few-shot classification Results](#Implemented Few-shot classification Results (will be gradually updated))
   - [miniImageNet results](#miniImageNet Dataset)
3. [General Guide](#General Guide)
   - [Installation](#installation)
   - [Runing a model](#running an implemented few-shot model)
   - [Creating new algorithms](#Creating a new few-shot algorithm)



## Advantages:
This repository builds on top of [LightningCLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html), which is very convenient to use after being familiar with this tool. 

1. Enabling multi-GPU training
   - Our implementation of FSL framework allows [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/notes/ddp.html) to be included in the training of Few-Shot Learning, which is not available before to the best of our knowledge. Previous researches use [DataParallel (DP)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) instead, which is inefficient and requires more computation storages. We achieve this by modifying the DDP sampler of Pytorch, making it possible to sample few-shot learning tasks among devices. See `dataset_and_process/samplers.py` for details.
2. High reimplemented accuracy
   - Our reimplementations of some FSL algorithms achieve strong performance. For example, our ResNet12 implementation of ProtoNet and Cosine Classifier achieves 76+ and 80+ accuracy on 5w5s task of miniImageNet, respectively. All results can be reimplemented using pre-defined configuration files in the config folder.
4. Quick and convenient creation of new algorithms
   - Pytorch-lightning provides our codebase with a clean and modular structure. Built on top of LightningCLI, our codebase unifies necessary basic components of FSL, making it easy to implement a brand-new algorithm. An impletation of an algorithm usually only requires three short additional files, one specifying the lightningModule, one specifying the classifer head, and the last one specifying all configurations. For example, see the code of ProtoNet (`modules/PN.py`, `architectures/classifier/proto_head.py`) and cosine classifier (`modules/cosine_classifier.py`, `architectures/classifier/CC_head.py`.
5. Enabling both episodic/non-episodic algorithms.

## Implemented Few-shot classification Results (will be gradually updated)
Implemented results on few-shot learning datasets with ResNet-12 backbone. The average results of 2,000 randomly sampled episodes repeated 5 times for 1/5-shot evaluation with 95% confidence interval are reported. 

### miniImageNet Dataset

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


## General Guide
To understand the code correctly, it is highly recommended to first quickly go through the [pytorch-lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/), especially [LightningCLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html). It won't be a long journey since pytorch-lightning is built on the top of pytorch.

### Installation
Just run the command:

```bash
pip install requirements.txt
```



### running an implemented few-shot model

Downloading Datasets (other datasets uploaded soon):

[miniImageNet](https://1drv.ms/u/s!AkYSH77Z8H6qa872NXTDnt-6bwY?e=XcKJgH)

Training (Except for Meta-baseline and COSOC):
    * Choose the corresponding configuration file in 'config'(e.g.`set_config_PN.py` for PN model), set  inside the parameter 'is_test' to False, set GPU ids (multi-GPU or not), dataset directory, logging dir as well as other parameters you would like to change.
    * modify the first line in run.sh (e.g., `python config/set_config_PN.py`).
    * To begin the running, run the command 
```bash
bash run.sh
```

Training Meta-baseline:
    * This is a two-stage algorithm, with the first stage being CEloss-pretraining, followed by ProtoNet finetuning. So a two-stage training is need. The first training uses the configuration file `config/set_config_meta_baseline_pretrain.py`. The second uses `config/set_config_meta_baseline_finetune.py`, with pre-training model path from the first stage, specified by the parameter`pre_trained_pathin` in the configuration file.

Training COSOC:
    * For pre-training Exemplar, choose configuration file `config/set_config_MoCo.py` and set parameter `is_exampler` to True.
    * For runing COS algorithm, run the command `python COS.py --[save_dir]`. `[save_dir]` specifies the saving directory of all foreground objects.
    * For runing a FSL algorithm with COS, choose configuration file `config/set_config_COSOC.py` and set parameter `data["train_dataset_params"]` to the directory of saved data of COS algorithm.

Testing:
    * Choose the same configuration file as training, set parameter `is_test` to True, `pre_trained_path` to the directory of checkpoint model (with suffix '.ckpt'), and other parameters (e.g. shot, batchsize) as you disire.
    * modify the first line in run.sh (e.g., `python config/set_config_PN.py`).
    * To begin the testing, run the command 
```bash
bash run.sh
```

### Creating a new few-shot algorithm

It is quite simple to implement your own algorithm. most of algorithms only need creation of a new LightningModule and a classifier head. 

#### run.py
The file run.py wraps the whole training and testing procedure of a FSL algorithm, for which all configurations are specified by an individual yaml file contained in the config folder; see set_config_PN.py for example. the file contains an python class Few_Shot_CLI, inherited from [LightningCLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html), adding new hyperpameters (Also specified in configuration file) as well as testing standard for FSL. It is usually not needed to modify this file.

#### FewShotModule
The folder `modules` contains LightningModules for FSL models, specifying model components, optimizers, logging metrics and train/val/test processes. Notably, `modules/base_module.py` contains the template module for all FSL models. All other modules inherit the base module; see `modules/PN.py` and `modules/cosine_classifier.py` for how episodic/non-episodic models inherit from the base module.

#### architectures
We divide general FSL architectures into feature extractor and classification head, specified respectively in `architectures/feature_extractor` and `architectures/classifier`. These are just common nn modules in pytorch. The recommended feature extractor is ResNet12, which is popular and shows promising performance. The classification head, however, varies with algorithms and need specific designs.

#### Datases and DataModule
Pytorch-lighning unifies data processing across training, val and testing into a single LightningDataModule. We disign such a datamodule in `dataset_and_process/datamodules/few_shot_datamodule.py` for FSL, enabling episodic/non-episodic sampling and DDP for multi-GPU fast training. The definition of Dataset itself is in `dataset_and_process/datasets`, specified as common pytorch datasets class. There is no need to modify the dataset module unless new datasets are involved.

#### callbacks and Plugins
See [documentation](https://pytorch-lightning.readthedocs.io/en/latest/) of pytorch-lightning for detailed introductions of callbacks and Plugins. They are additional functionalities added to the system in a modular fashion.













