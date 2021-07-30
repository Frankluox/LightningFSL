"""
Configurations for S2M2_R in paper "Charting the Right Manifold: Manifold Mixup for Few-shot Learning".
"""


from sacred import Experiment
import yaml
import time 
import os
ex = Experiment("S2M2_R", save_git_info=False)

@ex.config
def config():
    config_dict = {}

    #if training, set to False
    config_dict["load_pretrained"] = False
    #if training, set to False
    config_dict["is_test"] = False
    if config_dict["is_test"]:
        #if testing, specify the total rounds of testing. Default: 5
        config_dict["num_test"] = 5
        config_dict["load_pretrained"] = True
        #specify pretrained path for testing.
    if config_dict["load_pretrained"]:
        config_dict["pre_trained_path"] = "../results/CC/first_ex/version_2/checkpoints/epoch=58-step=17699.ckpt"
        #only load the backbone.
        config_dict["load_backbone_only"] = True

    #Specify the model name, which should match the name of file
    #that contains the LightningModule
    config_dict["model_name"] = "S2M2_R"
 
    

    #whether to use multiple GPUs
    multi_gpu = True
    if config_dict["is_test"]:
        multi_gpu = False
    #The seed
    seed = 10

    #The logging dirname: logdir/exp_name/
    log_dir = "../results/"
    exp_name = "S2M2/train"
    
    #Three components of a Lightning Running System
    trainer = {}
    data = {}
    model = {}


    ################trainer configuration###########################


    ###important###

    #debugging mode
    trainer["fast_dev_run"] = False
    trainer["limit_train_batches"] = 1.0
    if multi_gpu:
        trainer["accelerator"] = "ddp"
        trainer["sync_batchnorm"] = True
        trainer["gpus"] = [3,4,5,6]
    else:
        trainer["accelerator"] = None
        trainer["gpus"] = [0]
        trainer["sync_batchnorm"] = False
    
    # whether resume from a given checkpoint file
    trainer["resume_from_checkpoint"] = "../results/S2M2/train/version_3/checkpoints/last.ckpt" # example: "../results/ProtoNet/version_11/checkpoints/epoch=2-step=1499.ckpt"

    # The maximum epochs to run
    trainer["max_epochs"] = 600

    # potential functionalities added to the trainer.
    trainer["callbacks"] = [{"class_path": "pytorch_lightning.callbacks.LearningRateMonitor", 
                  "init_args": {"logging_interval": "step"}
                  },
                {"class_path": "callbacks.ModifiedModelCheckpoint",
                  "init_args":{"verbose": True, "save_last": True, "monitor": "val/acc", "mode": "max", "save_epochs": [99,199,299,399]}
                },
                {"class_path": "callbacks.SetSeedCallback",
                 "init_args":{"seed": seed, "is_DDP": multi_gpu}
                }]

    ###less important###
    num_gpus = trainer["gpus"] if isinstance(trainer["gpus"], int) else len(trainer["gpus"])
    trainer["logger"] = {"class_path":"pytorch_lightning.loggers.TensorBoardLogger",
                        "init_args": {"save_dir": log_dir,"name": exp_name}
                        }
    trainer["replace_sampler_ddp"] = False

    

    ##################shared model and datamodule configuration###########################

    #important
    test_shot = 5

    #less important
    per_gpu_val_batchsize = 1
    per_gpu_test_batchsize = 1
    way = 5
    val_shot = 5
    num_query = 15

    ##################datamodule configuration###########################

    #important

    #The name of dataset, which should match the name of file
    #that contains the datamodule.
    
    data["dataset_name"] = "miniImageNet"
    data["data_root"] = "../BF3S-master/data/mini_imagenet_split/images"
    #determine whether meta-learning.
    data["train_batchsize"] = 64
    
    data["train_num_workers"] = 8
    #the number of tasks
    data["val_num_task"] = 600
    data["test_num_task"] = 2000
    
    
    #less important
    data["num_gpus"] = num_gpus
    data["val_batchsize"] = num_gpus*per_gpu_val_batchsize
    data["test_batchsize"] = num_gpus*per_gpu_test_batchsize
    data["test_shot"] = test_shot
    data["val_num_workers"] = 8
    data["is_DDP"] = True if multi_gpu else False
    data["way"] = way
    data["val_shot"] = val_shot
    data["num_query"] = num_query
    data["drop_last"] = False
    data["is_meta"] = False

    ##################model configuration###########################

    #important

    #The name of feature extractor, which should match the name of file
    #that contains the model.
    model["backbone_name"] = "WRN_28_10"
    #the initial learning rate
    model["lr"] = 0.001


    #less important
    model["switch_epoch"] = 400
    model["alpha"] = 2.0
    model["scale_cls"] = 10.
    model["ft_batchsize"] = 4
    model["ft_epochs"] = 100
    model["ft_lr"] = 0.1
    model["ft_wd"] = 0.001
    model["way"] = way
    model["val_shot"] = val_shot
    model["test_shot"] = test_shot
    model["num_query"] = num_query
    model["val_batch_size_per_gpu"] = per_gpu_val_batchsize
    model["test_batch_size_per_gpu"] = per_gpu_test_batchsize
    model["weight_decay"] = 0
    #The name of optimization scheduler
    model["decay_scheduler"] = None
    model["optim_type"] = "adam"
    model["num_classes"] = 64

    

    config_dict["trainer"] = trainer
    config_dict["data"] = data
    config_dict["model"] = model



@ex.automain
def main(_config):
    config_dict = _config["config_dict"]
    file_ = 'config/config.yaml'
    stream = open(file_, 'w')
    yaml.safe_dump(config_dict, stream=stream,default_flow_style=False)

    