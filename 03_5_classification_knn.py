# Classification in machine learning
# Brett Melbourne
# 1 Feb 2024

# This example is from Chapter 2.2.3 of James et al. (2021). An Introduction to
# Statistical Learning. It is the simulated dataset in Fig 2.13.

import pandas as pd
import numpy as np
from plotnine import *
from itertools import product

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

# Orange-blue data:

orbludat = pd.read_csv("data/orangeblue.csv")

(ggplot(orbludat)
 + geom_point(aes(x="x1", y="x2", color="category"), shape="o", size=2)
 + scale_color_manual(values=["blue", "orange"])
 + theme(panel_grid=element_blank()))


# KNN function for a data frame of x_new
# x:       x data of variables in columns (dataframe, numeric)
# y:       y data, 2 categories (dataframe, character)
# x_new:   values of x variables at which to predict y (dataframe, numeric)
# k:       number of nearest neighbors to average (scalar, integer)
# rng:     numpy random generator, set ahead rng = np.random.default_rng()
#
# return:  predicted y at x_new (dataframe, character)
#
def knn_classify2(x, y, x_new, k, rng):
    x_new = np.array(x_new)
    n = len(x_new)
    df = pd.DataFrame()
    category = y.unique() #get the two category names
    df["y_int"] = np.where(y == category[0], 1, 0) #convert category to integer
    # Estimate probability of category 1 for each row of x_new
    p_cat1 = np.full(n, np.nan)
    for i in range(n):
    #   Distance of x_new to other x (Euclidean, i.e. sqrt(a^2+b^2+..))
        df["d"] = np.sqrt(np.sum((x - x_new[i])**2, axis=1))
    #   Sort y ascending by d; break ties randomly
        df["ran"] = rng.random(len(df))
        sorted_df = df.sort_values(by=["d","ran"])
    #   Mean of k nearest y data (frequency of category 1)
        p_cat1[i] = sorted_df[:k]["y_int"].mean()
    # Predict the categories
    y_pred = np.where(p_cat1 > 0.5, category[0], category[1])
    # Break ties if probability is equal (i.e. exactly 0.5)
    # rnd_category = rng.choice(category, n, replace=True) #vector of random labels
    # tol = 1 / (k * 10)  # tolerance for checking equality
    # y_pred = np.where(np.abs(p_cat1 - 0.5) < tol, rnd_category, y_pred)
    return pd.DataFrame(y_pred, columns=["category"])


# Test the output of the knn_classify2 function
rng = np.random.default_rng() #start random number generator
nm = pd.DataFrame(rng.random((4, 2)), columns=["x1","x2"])
knn_classify2(orbludat[['x1', 'x2']], orbludat['category'], nm, k=10, rng=rng)

# Plot
grid_x = expand_grid( {"x1": np.arange(0, 1.01, 0.01),
                       "x2": np.arange(0, 1.01, 0.01)})
pred_category = knn_classify2(x=orbludat[['x1', 'x2']], y=orbludat['category'],
                              x_new=grid_x, k=2, rng=rng)
preds = pd.concat([grid_x, pred_category], axis=1)

(ggplot(orbludat)
 + geom_point(aes(x="x1", y="x2", color="category"), shape="o", size=2)
 + geom_point(aes(x="x1", y="x2", color="category"), data=preds, size=0.1)
 + scale_color_manual(values=["blue", "orange"])
 + theme(panel_grid=element_blank()))



