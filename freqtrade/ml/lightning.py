from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Any, Optional
from wandb.wandb_run import Run
from datetime import datetime

import attr
import pandas as pd


@attr.s(repr=False)
class LightningConfig:
    """
    Configuration for LightningModule.
    Params
    -----
    :`name: str` - Model name
    :`timeframe: str` - example: "1m" or "30m" or "1h"
    :`exchange: str` - example: "binance"
    
    :`trainval_start: datetime` - example: "2021-03-01"
    :`trainval_end: datetime`   - example: "2021-05-26"
    
    :`opt_start: datetime`      - example: "2021-05-27"
    :`opt_end: datetime`        - example: "2021-06-27"
    
    :`test_start: datetime`     - example: "2021-06-28"
    :`test_end: datetime`       - example: "2021-07-28"
    
    Recommended Date Format
    - Date <= Last 2 months: Train and Val (trainval)
    - Date <= Last month   : Strategy optimization (opt)
    - Last Month to Present: Unbiased backtesting (test)
    """
    # User Defined Configuration
    name: str = attr.ib()
    timeframe: str = attr.ib()
    exchange: str = attr.ib()

    # Training and Val range
    trainval_start: datetime = attr.ib()
    trainval_end: datetime   = attr.ib()
    # Optimize profits in backtesting range
    opt_start: datetime      = attr.ib()
    opt_end: datetime        = attr.ib()
    # Unbiased backtesting range 
    test_start: datetime     = attr.ib()
    test_end: datetime       = attr.ib()
    
    # Optional configs
    num_training_epochs: Optional[int] = attr.ib(default=None)
    
    # DataFrame columns
    column_y: str = attr.ib(init=False, default="ml_label")
    # Updated after add_lables and before final_processing
    columns_unused: List[str] = attr.ib(init=False, default=["date", "open", "high", "low", "close", "volume", "pair", "ml_label"])
    columns_x: List[str] = attr.ib(init=False, default=None)
    
    # Late initialization
    data_filenames: List[str] = attr.ib(init=False, default=None)
    pairs: List[str] = attr.ib(init=False, default=None)
    
    def set_data_config(self, data_paths: List[Path]):
        """ Set data paths attribute. """
        self.data_filenames = [it.name for it in data_paths]
        self.pairs = [it.name.split("-")[0].replace("_", "/") for it in data_paths]

    def add_custom(self, key: str, value: Any):
        """ Add custom config attribute """
        if hasattr(self, key):
            raise Exception(f"Attribute with key '{key}' already exists!")
        
        setattr(self, key, value)


@attr.s(repr=False)
class LightningModule(ABC):
    """
    NOTE: Not a PyTorch Lightning module!
    TODO: Implement numpy version instead of dataframe to train NN.
    """    
    config: LightningConfig = None
    # Updated before and after training
    model: Any = None
    
    def __attrs_post_init__(self):
        self.config: LightningConfig = self.on_configure()
        self.config.set_data_config(
            data_paths = self.on_get_data_paths(Path.cwd(), self.config.timeframe, self.config.exchange)
        )
    
    @abstractmethod
    def on_configure(self) -> LightningConfig:
        """ List of Path to your per pair JSON data consisting of [date, open, high, low, close, volume] columns"""
        raise NotImplementedError()
    
    @abstractmethod
    def on_get_data_paths(self, cwd: Path, timeframe: str, exchange: str) -> List[Path]:
        """ List of Path to your per pair JSON data consisting of [date, open, high, low, close, volume] columns"""
        raise NotImplementedError()
    
    @abstractmethod
    def on_add_features(self, df_onepair: pd.DataFrame) -> pd.DataFrame:
        """ Define the features of dataset. Called after load data of this pair. 
        NOTE: Must import libraries that features are dependent on. E.G: `import talib.abstract as ta`, etc...
        This will be used when inferencing with freqtrade.
        ### Example
        ```python
        def add_features(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
            import talib.abstract as ta
            
            df_per_pair["ma_5"] = ta.SMA(df_per_pair, timeperiod=5)
            df_per_pair["rsi"] = ta.RSI(df_per_pair)
            
            return df_per_pair
        ```
        """
        raise NotImplementedError()
    
    @abstractmethod
    def on_add_labels(self, df_onepair: pd.DataFrame) -> pd.DataFrame:
        """ Define the label (target prediction) of per pair data. Called after add_features() 
        NOTE: The label must in column named `self.config.column_y`!
        ### Example
        ```python
        def add_labels(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
            # Classify future returns into 5 classes
            df_per_pair[self.config.column_y] = pd.qcut(df_per_pair["future_returns"], 5, labels=False)
            return df_per_pair
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def on_final_processing(self, df_allpairs: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        """ Final processing step for the combined data (all pairs). Called after add_labels()
        Returns: (X_train, X_val, y_train, y_val)
        ### Example
        ```python
        def final_processing(self, df_allpairs: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
            X = df_allpairs[self.config.columns_x]
            y = df_allpairs[self.config.column_y]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
            return X_train, X_val, y_train, y_val
        ```
        """
        raise NotImplementedError()
    
    @abstractmethod
    def on_define_model(self, run: Run, X_train, X_val, y_train, y_val) -> Any:
        """ Define your untrained model. Then you can refer to self.MODEL 
        Returns: model object
        ### Example
        ```python
        def define_model(self, run: Run, X_train, X_val, y_train, y_val) -> Any:
            return RandomForestClassifier(max_depth=2, random_state=0)
        ```
        """
        raise NotImplementedError()
    
    @abstractmethod
    def on_start_training(self, run: Run, X_train, X_val, y_train, y_val):
        """ Start model training with wandb.init() as run.
        NOTE: Refer to self.model when training!
        ### Simple Example
        ```python
        def start_training(self, run: Run, X_train, X_val, y_train, y_val):
            self.model: RandomForestClassifier
            self.model.fit(X_train, y_train)
        ```
        Tips:
        - Save model checkpoint as wandb artifact every 100s of epochs or so.
        - When the training ends, save the final model.
        - It is recommended to log loss, accuracy, etc. using self.training_step()
        - After training, to refer your trained model, you can refer to self.model to evaluate.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def on_predict(self, df_input_perpair: pd.DataFrame) -> pd.DataFrame:
        """ Returns the DataFrame of prediction. This inference will be
            used in strategy.py file loaded from wandb. The returning
            DataFrame results may contain multiple columns and modifies the
            original DataFrame.
        """
        raise NotImplementedError()
    
    def on_training_step(self, run: Run, data: dict):
        """ Called every training step to log process. """
        raise NotImplementedError()
        
        