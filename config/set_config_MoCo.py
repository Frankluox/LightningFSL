"""
Training MoCo---Momentum Contrast for Unsupervised Visual Representation Learning, CVPR 2020,
or Exampler---What makes instance discrimination good for transfer learning? ICLR 2021.
"""


from sacred import Experiment
import yaml
import time 
import os
ex = Experiment("MoCo", save_git_info=False)

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
        config_dict["pre_trained_path"] = "../results/CC/version_11/checkpoints/epoch=52-step=26499.ckpt"
        #only load the backbone.
        config_dict["load_backbone_only"] = False

    #Specify the model name, which should match the name of file
    #that contains the LightningModule
    config_dict["model_name"] = "moco"
 
    

    #whether to use multiple GPUs
    multi_gpu = True
    if config_dict["is_test"]:
        multi_gpu = False
    #The seed
    seed = 10

    #The logging dirname: logdir/exp_name/
    log_dir = "../results/"
    exp_name = "MoCo/first_ex"
    
    #Three components of a Lightning Running System
    trainer = {}
    data = {}
    model = {}


    ################trainer configuration###########################


    ###important###

    #debugging mode
    trainer["fast_dev_run"] = True

    if multi_gpu:
        trainer["accelerator"] = "ddp"
        trainer["sync_batchnorm"] = True
        trainer["gpus"] = [1,2]
        trainer["plugins"] = [{"class_path": "plugins.modified_DDPPlugin"}]
    else:
        trainer["accelerator"] = None
        trainer["gpus"] = [2]
        trainer["sync_batchnorm"] = False
    
    # whether resume from a given checkpoint file
    trainer["resume_from_checkpoint"] = None # example: "../results/ProtoNet/version_11/checkpoints/epoch=2-step=1499.ckpt"

    # The maximum epochs to run
    trainer["max_epochs"] = 1000

    
    # potential functionalities added to the trainer.
    trainer["callbacks"] = [{"class_path": "pytorch_lightning.callbacks.LearningRateMonitor", 
                  "init_args": {"logging_interval": "step"}
                  },
                {"class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                  "init_args":{"verbose": True, "save_last": True, "monitor": "val/acc", "mode": "max"}
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
    per_gpu_val_batchsize = 8
    per_gpu_test_batchsize = 8
    way = 5
    val_shot = 5
    num_query = 15

    ##################datamodule configuration###########################

    #important

    #The name of dataset, which should match the name of file
    #that contains the datamodule.
    
    data["train_dataset_name"] = "miniImageNet_contra"

    data["train_data_root"] = "../BF3S-master/data/mini_imagenet_split/images"

    data["val_test_dataset_name"] = "miniImageNet"

    data["val_test_data_root"] = "../BF3S-master/data/mini_imagenet_split/images"



    data["train_batchsize"] = 256
    data["train_num_workers"] = 8
    #the number of tasks
    data["val_num_task"] = 1200
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
    model["backbone_name"] = "resnet12"
    model["is_Exampler"] = True
    #the initial learning rate
    model["lr"] = 0.1*data["train_batchsize"]/256


    #less important
    model["momemtum"] = 0.999
    model["temparature"] = 0.07
    model["queue_len"] = 65536
    model["mlp_dim"] = 128
    
    model["is_DDP"] = multi_gpu
    model["task_classifier_name"] = "proto_head"
    model["task_classifier_params"] = {"learn_scale":False}
    model["way"] = way
    model["val_shot"] = val_shot
    model["test_shot"] = test_shot
    model["num_query"] = num_query
    model["val_batch_size_per_gpu"] = per_gpu_val_batchsize
    model["test_batch_size_per_gpu"] = per_gpu_test_batchsize
    model["weight_decay"] = 5e-4
    #The name of optimization scheduler
    model["decay_scheduler"] = "cosine"
    model["optim_type"] = "sgd"

    

    config_dict["trainer"] = trainer
    config_dict["data"] = data
    config_dict["model"] = model



@ex.automain
def main(_config):
    config_dict = _config["config_dict"]
    file_ = 'config/config.yaml'
    stream = open(file_, 'w')
    yaml.safe_dump(config_dict, stream=stream,default_flow_style=False)

    