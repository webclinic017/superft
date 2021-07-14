import pandas as pd

# Given dictionary data
data = {
    "mean": [0.1],
    "median": [0.5],
    "std": [0.15],
    "max":[0.8],
    "min": [0.1]
}

# Create a Pandas dataframe
print(pd.DataFrame(data).T)