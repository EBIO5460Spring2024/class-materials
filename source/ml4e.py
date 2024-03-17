from itertools import product
import pandas as pd

# Create a data frame with a grid of all combinations of the variables specified
# in grid_dict.
#     https://pandas.pydata.org/pandas-docs/version/0.17.1/
#     cookbook.html#creating-example-data
# grid_dict: the variables to make the grid (dictionary)
# return: combinations (dataframe)
#
def expand_grid(grid_dict):
    rows = product(*grid_dict.values())
    return pd.DataFrame.from_records(rows, columns=grid_dict.keys())
