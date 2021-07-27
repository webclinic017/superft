from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Any
from wandb.wandb_run import Run

import attr
import pandas as pd


@attr.s
class TradingLightningModule(ABC):
    """
    NOTE: Not a PyTorch Lightning module!
    TODO: Define a numpy version instead of dataframe.
    """
    # User Defined Configuration
    NAME: str = attr.ib()
    TIMEFRAME: str = attr.ib()
    EXCHANGE: str = attr.ib()
    COLUMNS_NOT_USE: List[str] = attr.ib(default=["date", "open", "high", "low", "close", "volume", "pair", "ml_label"])
    """
    Data Format
    - Date <= Last 2 months: Train and Val (Shuffled val data)
    - Date <= Last month   : Strategy optimization
    - Last Month to Present: Unbiased backtesting
    """
    # Training and Val range
    TRAINVAL_START = attr.ib(default="2021-03-01")
    TRAINVAL_END   = attr.ib(default="2021-05-26")
    # Optimize profits in backtesting range
    OPT_START      = attr.ib(default="2021-05-27")
    OPT_END        = attr.ib(default="2021-06-27")
    # Unbiased backtesting range 
    TEST_START     = attr.ib(default="2021-06-28")
    TEST_END       = attr.ib(default="2021-07-28")
    
    # Updated after training
    MODEL = None
    
    def __attrs_post_init__(self):
        self.DATA_PATHS = [str(it) for it in self.get_data_paths()]
        self.PAIRS = [it.name.split("-")[0].replace("_", "/") for it in self.get_data_paths()]
    
    @abstractmethod
    def get_data_paths(self) -> List[Path]:
        """ List of Path to your per pair JSON data consisting of date, open, high, low, close, volume"""
        raise NotImplementedError()
    
    @abstractmethod
    def add_features(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        """ Define the features of dataset. Called after load data of this pair. 
            NOTE: Must import libraries that features are dependent on. 
            E.G: import talib.abstract as ta, etc...
        """
        raise NotImplementedError()
    
    @abstractmethod
    def add_labels(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        """ Define the label (target prediction) of per pair data. Called after add_features() 
            NOTE: The label must in column named "ml_label"!
        """
        raise NotImplementedError()

    @abstractmethod
    def post_processing(self, df_combined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Extra processing step for the combined data (all pairs). Called after add_labels()
            Returns: (Train DataFrame, Val DataFrame)
        """
        raise NotImplementedError()
    
    @abstractmethod
    def define_model(self, run: Run, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Any:
        """ Define your untrained model. Then you can refer to self.MODEL """
        raise NotImplementedError()
    
    @abstractmethod
    def start_training(self, run: Run, df_train: pd.DataFrame, df_val: pd.DataFrame):
        """ Start model training with wandb.init() as run.
            NOTE: Please call self.set_untrained_model(model) before training!
            Tips:
            - Save model checkpoint as wandb artifact every 100s of epochs or so.
            - After training ends, save the final model.
            - It is recommended to log loss, accuracy, etc. using self.training_step()
            - After training to refer your trained model, you can refer to self.MODEL
        """
        raise NotImplementedError()
    
    def training_step(self, run: Run, data: dict):
        """ Called every training step to log process. """
        raise NotImplementedError()
    
    def _pre_training(self, run: Run):
        """ Pre training checks """
    
    def _post_training(self, run: Run):
        """ Post training checks """
        