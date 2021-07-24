from importlib import import_module
from .base_module import BaseFewShotModule
def get_module(module_name):
    model_module = import_module("." + module_name, package="modules")
    model_module = getattr(model_module, "get_model")
    return model_module()