from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser
from modules import get_module, BaseFewShotModule
from dataset_and_process import FewShotDataModule
class Few_Shot_CLI(LightningCLI):
    def __init__(self,**kwargs) -> None:
        """Add two config parameters:
             --is_test: determine the mode
             --model_name: The few-shot model name. For example, PN.
        Args:
            kwargs: Original parameters of LightningCLI
        """
        super().__init__(**kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            'is_test',
            type=bool,
            default=False,
            help="whether in testing only mode"
        )
        parser.add_argument(
            'model_name',
            type=str,
            default="",
            help="The model name to train on.\
                  It should match the file name that contains the model."
        )
    def parse_arguments(self) -> None:
        """Parses command line arguments and stores it in self.config"""
        self.config = self.parser.parse_args(_skip_check = True)
    
    def before_instantiate_classes(self) -> None:
        self.model_class = get_module(self.config["model_name"])
        self.is_test = self.config["is_test"]

    def fit(self):
        """Runs fit of the instantiated trainer class and prepared fit keyword arguments"""
        if self.is_test:
            pass
        else:
            self.trainer.fit(**self.fit_kwargs)
    def after_fit(self):
        """Runs testing"""
        if self.is_test:
            self.trainer.test(self.model, datamodule=self.datamodule)
        else:
            pass

if __name__ == '__main__':
    cli = Few_Shot_CLI(
        model_class= BaseFewShotModule, 
        datamodule_class = FewShotDataModule, 
        seed_everything_default=1234
    )