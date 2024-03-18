### Assignment 4

**Due:** Thursday 28 Mar 11:59 PM

**Grading criteria:** Answer all the questions completely. On time submission.

**Percent of grade:** 14%




#### Boosting and neural networks
**Learning goals:**

* Use state of the art boosting software with typical ecological data
* Explore parameters of the xgboost algorithm
* Understand the model algorithm of a basic neural network
* Practice modifying a function to examine its internal workings
* Practice tuning machine learning algorithms using the CV inference algorithm
* Practice comparing model accuracy using the CV inference algorithm
* Practice visualizing model outputs



#### Part 1: Boosting

Your goal is to train and tune a gradient boosting model to predict the occurrence of a plant in New Zealand based on a range of predictor variables. The plant is "nz05" from an anonymized reference dataset (apparently NZ plants have better data-privacy protections than US citizens). These data are from one of the papers we'll read: 

* Valavi et al. (2021). Predictive performance of presence-only species distribution models: a benchmark study with reproducible code. *Ecological Monographs* 0:e01486. https://doi.org/10.1002/ecm.1486.

This could be super fun to expand on later in an individual project to see if you can improve the predictions in this paper.

```R
library(disdat) #data package
library(dplyr)
library(ggplot2)
library(sf)
library(xgboost)
```

Load presence-absence data for species "nz05"

```R
nz05df <- bind_cols(select(disPa("NZ"), nz05), disEnv("NZ")) |> 
    rename(occ=nz05)
head(nz05df)
```

Outline of New Zealand

```R
nzpoly <- disBorder("NZ")
class(nzpoly) #sf = simple features; common geospatial format
```

Plot presence (1) absence (0) data

```R
nz05df |> 
    arrange(occ) |> #place presences on top
    ggplot() +
    geom_sf(data=nzpoly, fill="lightgray") +
    geom_point(aes(x=x, y=y, col=factor(occ)), shape=1, alpha=0.2) +
    theme_void()
```

Data for modeling

```R
nz05pa <- nz05df |> 
    select(!c(group,siteid,x,y,toxicats))
head(nz05pa)
```

Example boosted model (30-60 secs)

```R
# Prepare data for xgboost
nz05pa_xgb <- xgb.DMatrix(data=as.matrix(nz05pa[,-1]), label=nz05pa$occ)
class(nz05pa_xgb)
colnames(nz05pa_xgb)

# Train
nz05_train <- xgboost(data=nz05pa_xgb, max.depth=1, eta=0.01, nthread=2, 
                      nrounds=10000, print_every_n=1000, 
                      objective="binary:logistic")

# Predict
nz05_prob <- predict(nz05_train, newdata=nz05pa_xgb)
nz05_pred <- 1 * (nz05_prob > 0.5)

# Characteristics of this prediction
hist(nz05_prob)
max(nz05_prob)
sum(nz05_prob > 0.5) #number of predicted presences

table(nz05_pred, nz05pa$occ)  #confusion matrix
mean(nz05_pred == nz05pa$occ) #accuracy
mean(nz05_pred != nz05pa$occ) #error = 1 - accuracy
```

Example prediction for a grid of the predictor variables across NZ. The grid was prepared from the raster (geotiff) files associated with the paper, downloaded from https://osf.io/kwc4v/files/.

```R
# Read in the grid of predictor variables
NZ_grid <- read.csv("data/NZ_predictors.csv")
head(NZ_grid)

# Prepare data for xgboost
NZ_grid_xgb <- xgb.DMatrix(data=as.matrix(select(NZ_grid, !c(x,y))))
colnames(NZ_grid_xgb)

#Predict
nz05_grid_prob <- predict(nz05_train, newdata=NZ_grid_xgb)
nz05_grid_present <- 1 * (nz05_grid_prob > 0.5)

# Map probability prediction
NZ_grid |>
    bind_cols(prob=nz05_grid_prob) |>
    ggplot() +
    geom_tile(aes(x=x, y=y, fill=prob)) +
    scale_fill_viridis_c() +
    coord_equal() +
    theme_void() +
    labs(fill = "Probability")
    
# Map presence prediction
NZ_grid |>
    bind_cols(present=nz05_grid_present) |>
    ggplot() +
    geom_tile(aes(x=x, y=y, fill=factor(present))) +
    coord_equal() +
    theme_void() +
    labs(fill = "Present")
```



**Q1\.** Train and tune a gradient boosting model to find the best predictive performance across the five boosting parameters ( `eta`, `max_depth`, `subsample`, `colsample_bytree`, `nrounds`). 

Suggestions:

* Use a grid search
* Use 5-fold CV
* Use parallel processing, which in `xgboost` is best done using `xgboost` itself. This is super easy. Just set the `nthread` argument to a number greater than 1. I suggest using as many cores as your computer has, or even better, use a supercomputer node. Higher-level parallel processing on the grid search is not helpful (according to xgboost documentation).
* Assess models using the error rate, i.e.
  * `mean(nz05_pred != nz05pa$occ)`

**Q2\.** Like other tree ensemble methods we have looked at, `xgboost` can provide information about the relative importance of the different predictor variables. Research how to do this and make a plot that displays this information.

**Q3\.** Plot the prediction both as a presence map and a map of probabilities.



#### Part 2: Understanding neural networks

**Q3\.** Code a network by hand. Extend the "by-hand" code in [07_3_ants_neural_net.R](07_3_ants_neural_net.R) to a 2-layer neural network, each layer with 5 nodes. Plot the model predictions. Use these weights and biases (n.b. you don't need to train the model):

```R
w1 <- c(-1.0583408, -0.6441127,  1.1663090,  0.08298533, -0.41105017,
        -0.8540244,  0.5407082, -0.2184951, -0.11781270,  0.36039100,
         0.8608801,  1.2520101, -0.1495921,  0.83325340,  1.15322390,
        -0.5394921, -0.7117111,  0.1879681,  0.30929375,  0.05233159) |>
    matrix(nrow=4, ncol=5, byrow=TRUE)

b1 <- c(0.1909237, 0.5486836, -0.1032256, 0.6253318, 0.2843419)

w2 <- c(0.04039513, 0.7977440,  0.60440171, -0.1800931, -0.210001990,
       -0.14771833, 0.3682977,  0.95937222,  0.3446860,  0.008643006,
       -0.34225080, 1.2922773,  0.11651120,  0.5326685, -0.592227300,
       -0.79168826, 0.5419835, -0.05803596, -1.2168059,  0.169808860,
        0.43390460, 1.0874641,  0.54609700,  0.2390731, -0.599693800) |>
    matrix(nrow=5, ncol=5, byrow=TRUE)

b2 <- c(-0.29183790, 0.32845289, 0.32393071, 0.06806916, -0.01153159)

w3 <- c(-0.3925169, 0.8072395, 1.398517, -0.7064973, -0.3754095) |>
    matrix(nrow=5, ncol=1, byrow=TRUE)

b3 <- 0.3231535
```

Compare plots of predictions for this 2-layer model to the single-layer model. Describe qualitatively (i.e. make a comment) how the predictions differ.

Hint: don't start from scratch, just add a few lines of code here and there where needed. The goal of this is to gain greater understanding of the algorithm.



#### Part 3. Neural networks in practical use

First, get your computer setup with tensorflow and keras. See [07_6_install_keras.md](07_6_install_keras.md) for installing on your local machine (Windows or MacOS), or [08_6_CU_supercomputer.md](08_6_CU_supercomputer.md) for getting set up on the supercomputer.

**Q4\.** Using keras, train and tune a neural network to compare to the boosting algorithm you used above for plant species "nz05".

Suggestions:

* Use a feedforward network
* Use binary_crossentropy loss
* Tuning: we don't have time to try many combinations or to do k-fold CV (you would do this for a research project). Here is a strategy:
  * use the cross validation option built into keras (i.e. `fit()` argument `validation_split=0.2`)
  * try four different architectures, e.g. 25 wide, 50 wide, 5x5 deep, 5x10 deep
  * try adding dropout regularization to the layers with 0.3 as a default rate
* Early stopping: often it is advantageous to stop learning after some number of epochs to prevent overfitting (i.e. when you see the validation error start to go back up)

**Q5\.** Compare to the boosting model. Which model gave the best predictive performance?

* We can't formally compare the models here. We would need to first set aside a test set to make the comparison, i.e. we'd need a three-way split: train-validate-test, only comparing the models using the test set after tuning the models on the validation set.
* Nevertheless, does the neural network get within the ballpark of the boosting model comparing the k-fold CV mean accuracy of the boosting model to the CV accuracy of the neural network?

**Q6\.** Plot predictions as you did for the boosting model.

