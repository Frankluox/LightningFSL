from importlib import import_module

def get_dataset(dataset_name: str):
    r"""Import a dataset given the name of file that contains it.

    Args:
        dataset_name: the name file that contains wanted dataset
    """
    dataset_module = import_module("." + dataset_name, package="dataset_and_process.datasets")
    return_class =  getattr(dataset_module, "return_class")
    return return_class()