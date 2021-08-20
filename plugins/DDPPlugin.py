from pytorch_lightning.plugins.training_type import DDPPlugin

class modified_DDPPlugin(DDPPlugin):
    """Modify DDPPlugin, making parameter "find_unused_parameters" explicit.
       Turn down "find_unused_parameters" if possible, in order to improve performance.
    """
    def __init__(self, 
        find_unused_parameters: bool = False,
        parallel_devices = None,
        num_nodes = None,
        cluster_environment = None,
        sync_batchnorm = None,
        ddp_comm_state = None,
        ddp_comm_hook = None,
        ddp_comm_wrapper = None,
        **kwargs):

        super().__init__(parallel_devices, num_nodes, cluster_environment,
                         sync_batchnorm, ddp_comm_state, ddp_comm_hook, ddp_comm_wrapper, **kwargs)
        self._ddp_kwargs["find_unused_parameters"] = find_unused_parameters

    
if __name__ == "__main__":
    a = modified_DDPPlugin()