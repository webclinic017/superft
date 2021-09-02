from joblib import Parallel, delayed, parallel_backend

import os
import pandas as pd
import numpy as np
import time


def df_processing_function(df_inp):
    for i in range(100):
        df_inp[f"ma_{i}"] = df_inp["close"].rolling(i).mean()
    return df_inp


def parallelize_dataframe(df_input, func, split_column: str, n_cores=4):
    df_split = [df_input[df_input[split_column] == category] for category in df_input[split_column].unique()]
    result = Parallel(prefer="threads", n_jobs=n_cores)(delayed(func)(_df.copy()) for _df in df_split)
    return pd.concat(result)


def parallelize_dataframe_mp(df_input, func, split_column: str, n_cores=4):
    df_split = deque([df_input[df_input[split_column] == category] for category in df_input[split_column].unique()])
    pool = Pool(n_cores)
    
    full_result = []
    
    while len(df_split) > 0:
        batch = [df_split.pop() for _ in range(n_cores) if len(df_split) > 0]
        full_result.extend(pool.map(func, batch))
    
    df = pd.concat(full_result)
    pool.close()
    pool.join()
    return df


def vanilla(df_input, func, split_column: str):
    df_split = [df_input[df_input[split_column] == category] for category in df_input[split_column].unique()]
    return pd.concat([func(it) for it in df_split])


if __name__ == "__main__":
    os.environ["MODIN_ENGINE"] = "dask"
    from distributed import Client
    client = Client(n_workers=16)
    import modin.pandas as mpd
    import pandas as pd
    
    df = pd.DataFrame({"pair": [], "close": []})

    for pair in ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "BCH/USDT", "DAI/USDT", "ETC/USDT"]:
        d1 = pd.DataFrame({"close": [np.random.randn() for _ in range(100000)],})
        d1["pair"] = pair
        df = pd.concat([df, d1], axis=0)
    
    mdf = mpd.DataFrame(df)
    
    t0 = time.time()
    vanilla(mdf, df_processing_function, "pair")
    print(time.time() - t0)
    import pandas_ta