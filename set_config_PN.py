from sacred import Experiment
import yaml
import time 
import os
ex = Experiment("ProtoNet", save_git_info=False)

@ex.config
def config():
    config_dict = {}

    config_dict["is_test"] = False
    config_dict["model_name"] = "PN"
    multi_gpu = False

    
    trainer = {}
    data = {}
    model = {}


    ################trainer configuration###########################
    if multi_gpu:
        trainer["accelerator"] = "ddp"
        trainer["sync_batchnorm"] = True
        trainer["gpus"] = 2
    else:
        trainer["accelerator"] = None
        trainer["gpus"] = [4]
        trainer["sync_batchnorm"] = False

    trainer["fast_dev_run"] = False
    log_dir = "../results/"
    exp_name = "ProtoNet"
    trainer["logger"] = {"class_path":"pytorch_lightning.loggers.TensorBoardLogger",
                        "init_args": {"save_dir": log_dir,"name": exp_name}
                        }
    trainer["max_epochs"] = 60
    trainer["replace_sampler_ddp"] = False

    ##################shared model and datamodule configuration###########################

    #important
    per_gpu_train_batchsize = 2
    train_shot = 5
    test_shot = 5

    #less important
    per_gpu_val_batchsize = 8
    per_gpu_test_batchsize = 8
    way = 5
    val_shot = 5
    num_query = 15

    ##################datamodule configuration###########################

    #important
    data["dataset_name"] = "miniImageNet"
    data["data_root"] = ""
    data["is_meta"] = True
    
    
    #less important
    data["per_gpu_train_batchsize"] = per_gpu_train_batchsize
    data["per_gpu_val_batchsize"] = per_gpu_val_batchsize
    data["per_gpu_test_batchsize"] = per_gpu_test_batchsize
    data["test_shot"] = test_shot
    data["train_shot"] = train_shot
    data["train_num_workers"] = 4
    data["val_num_workers"] = 4
    data["is_DDP"] = True if multi_gpu else False
    data["train_num_task_per_epoch"] = 1000
    data["val_num_task"] = 1200
    data["test_num_task"] = 1200
    data["way"] = way
    data["val_shot"] = val_shot
    data["num_query"] = num_query
    data["drop_last"] = False

    ##################model configuration###########################

    #important
    model["backbone_name"] = "resnet12"
    model["lr"] = 0.1 if multi_gpu else 0.05


    #less important
    model["way"] = way
    model["train_shot"] = train_shot
    model["val_shot"] = val_shot
    model["test_shot"] = test_shot
    model["num_query"] = num_query
    model["train_batch_size_per_gpu"] = per_gpu_train_batchsize
    model["val_batch_size_per_gpu"] = per_gpu_val_batchsize
    model["test_batch_size_per_gpu"] = per_gpu_test_batchsize
    model["weight_decay"] = 5e-4
    model["decay_scheduler"] = "cosine"
    model["optim_type"] = "sgd"
    model["metric"] = "cosine"
    model["scale_cls"] = 10.
    

    config_dict["trainer"] = trainer
    config_dict["data"] = data
    config_dict["model"] = model



@ex.automain
def main(_config):
    config_dict = _config["config_dict"]
    file_ = 'config.yaml'
    stream = open(file_, 'w')
    yaml.safe_dump(config_dict, stream=stream,default_flow_style=False)

    