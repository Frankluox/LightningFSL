from sacred import Experiment
import yaml
import time 
import os
ex = Experiment("COSOC", save_git_info=False)

@ex.config
def config():
    config_dict = {}

    # whether load pretrained model. This is is different from resume_from_checkpoint that loads everything from a breakpoint.
    config_dict["load_pretrained"] = True
    #if training, set to False
    config_dict["is_test"] = False
    if config_dict["is_test"]:
        #if testing, specify the total rounds of testing. Default: 5
        config_dict["num_test"] = 5
        config_dict["load_pretrained"] = True
        #specify pretrained path for testing.
    if config_dict["load_pretrained"]:
        config_dict["pre_trained_path"] = "../meta_learning_framework/results/miniImageNet/res12_PN_prob_crop/04_30_15_43_finetune_save_all_3crop_alpha0.5/encoder_net_epoch16.best"
        # config_dict["pre_trained_path"] = "../results/Exemplar/1000epoch/version_0/checkpoints/epoch=864-step=129749.ckpt"
        # config_dict["pre_trained_path"] = "../results/ProtoNet/test1gpu/version_0/checkpoints/epoch=59-step=29999.ckpt"
        #only load the backbone.
        config_dict["load_backbone_only"] = True

    #Specify the model name, which should match the name of file
    #that contains the LightningModule
    config_dict["model_name"] = "COSOC"

    

    #whether to use multiple GPUs
    multi_gpu = False
    if config_dict["is_test"]:
        multi_gpu = False
    #The seed
    seed = 10
    config_dict["seed"] = seed

    #The logging dirname: logdir/exp_name/
    log_dir = "../results/"
    exp_name = "COSOC/test"
    
    #Three components of a Lightning Running System
    trainer = {}
    data = {}
    model = {}


    ################trainer configuration###########################


    ###important###

    #debugging mode
    trainer["fast_dev_run"] = False

    if multi_gpu:
        trainer["accelerator"] = "ddp"
        trainer["sync_batchnorm"] = True
        trainer["gpus"] = [2,3]
        trainer["plugins"] = [{"class_path": "plugins.modified_DDPPlugin"}]
    else:
        trainer["accelerator"] = None
        trainer["gpus"] = [2]
        trainer["sync_batchnorm"] = False
    
    # whether resume from a given checkpoint file
    trainer["resume_from_checkpoint"] = None # example: "../results/ProtoNet/version_11/checkpoints/epoch=2-step=1499.ckpt"

    # The maximum epochs to run
    trainer["max_epochs"] = 20

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
    test_shot = 1
    num_patch = 7
    #less important
    per_gpu_val_batchsize = 1
    per_gpu_test_batchsize = 1
    way = 5
    val_shot = 1
    num_query = 15

    ##################datamodule configuration###########################

    #important

    #The name of dataset, which should match the name of file
    #that contains the datamodule.
    data["train_dataset_name"] = "miniImageNet_prob_crop"
    data["val_test_dataset_name"] = "miniImageNet_multi_crop"
    data["train_dataset_params"] = {"feature_image_and_crop_id":"../result/feature_image_and_crop_id.pkl","position_list":"../result/position_list.npy"}
    data["train_data_root"] = "../BF3S-master/data/mini_imagenet_split/images"
    data["val_test_data_root"] = "../BF3S-master/data/mini_imagenet_split/images"
    data["val_test_dataset_params"] = {"num_patch":num_patch}
    #determine whether meta-learning.
    data["train_batchsize"] = 128
    
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
    model["backbone_name"] = "resnet12"
    #the initial learning rate
    model["lr"] = 0.005*data["train_batchsize"]/128


    #less important
    model["SOC_params"] = {"num_patch": num_patch, "alpha": 1.0, "beta": 0.8}
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

    