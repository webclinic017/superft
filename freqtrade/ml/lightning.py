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
    TODO: Implement numpy version instead of dataframe to train NN.
    """
    # User Defined Configuration
    name: str = attr.ib()
    timeframe: str = attr.ib()
    exchange: str = attr.ib()
    columns_unused: List[str] = attr.ib(default=["date", "open", "high", "low", "close", "volume", "pair", "ml_label"])
    """
    Data Format
    - Date <= Last 2 months: Train and Val (Shuffled val data)
    - Date <= Last month   : Strategy optimization
    - Last Month to Present: Unbiased backtesting
    """
    # Training and Val range
    trainval_start = attr.ib(default="2021-03-01")
    trainval_end   = attr.ib(default="2021-05-26")
    # Optimize profits in backtesting range
    opt_start      = attr.ib(default="2021-05-27")
    opt_end        = attr.ib(default="2021-06-27")
    # Unbiased backtesting range 
    test_start     = attr.ib(default="2021-06-28")
    test_end       = attr.ib(default="2021-07-28")
    
    # Updated after training
    model = None
    
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
            This will be used when inferencing with freqtrade.
            E.G: `import talib.abstract as ta`, etc...
        """
        raise NotImplementedError()
    
    @abstractmethod
    def add_labels(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        """ Define the label (target prediction) of per pair data. Called after add_features() 
            NOTE: The label must in column named "ml_label"!
        """
        raise NotImplementedError()

    @abstractmethod
    def final_processing(self, df_combined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Final processing step for the combined data (all pairs). Called after add_labels()
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
            NOTE: Refer to self.model when training!
            Tips:
            - Save model checkpoint as wandb artifact every 100s of epochs or so.
            - When the training ends, save the final model.
            - It is recommended to log loss, accuracy, etc. using self.training_step()
            - After training, to refer your trained model, you can refer to self.model to evaluate.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, df_input: pd.DataFrame):
        """ Returns the Series of prediction. This inference will be
            used in strategy.py file loaded from wandb.
        
        1. Check if feature columns and datatypes are supported.
           If not, create new df, convert dtype, then call add_features()
           
        NOTE: This method must NOT change the existing input dataframe.
        """
        raise NotImplementedError()
    
    def training_step(self, run: Run, data: dict):
        """ Called every training step to log process. """
        raise NotImplementedError()
    
    def _pre_training(self, run: Run):
        """ Pre training checks """
        raise NotImplementedError()
    
    def _post_training(self, run: Run):
        """ Post training checks """
        raise NotImplementedError()
        