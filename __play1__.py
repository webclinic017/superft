import pandas as pd


df = pd.DataFrame({
    "a": [1,2,3],
    "b": [3,2,1],
})

for i in df.itertuples():
    print(i.a)