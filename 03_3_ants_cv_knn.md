Ant data: k nearest neighbors model algorithm
================
Brett Melbourne
29 Jan 2024

KNN for the regression case illustrated with the ants data. This code is
much the same as `ants_cv_polynomial.R` but instead of using a
polynomial as the model algorithm we use KNN.

``` r
library(ggplot2)
library(dplyr)
library(tidyr)
```

Forest ant data:

``` r
forest_ants <- read.csv("data/ants.csv") |> 
    filter(habitat=="forest")
```

K Nearest Neighbors (KNN) algorithm for 1 new value of x, translating
our pseudocode to R code.

``` r
# Set k = number of nearest neighbors
# Input (x, y) = x, y data pairs
# Input x_new = x value at which to predict y_new
# Calculate d = distance of x_new to other x 
# Sort y data ascending by d; break ties randomly
# Predict new y = mean of k nearest y data
```
