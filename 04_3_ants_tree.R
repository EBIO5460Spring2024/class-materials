#' ---
#' title: "Ant data: decision trees"
#' author: Brett Melbourne
#' date: 6 Feb 2024
#' output:
#'     github_document
#' ---

#' Decision trees for the regression case illustrated with the ants data. We
#' start with a regression tree for a single predictor variable, then look at
#' regression trees with multiple predictors.

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
library(tree)
source("source/tree_helper_functions.R")

#' Forest ant data:

forest_ants <- read.csv("data/ants.csv") |> 
    filter(habitat=="forest") |> 
    select(latitude, richness)

#' Before we look at the algorithms in detail, we'll start by using the `tree()`
#' function in the `tree` package. Using this package we'll first train the
#' model, visualize the regression tree, and use it to make predictions of
#' species richness.

tree_trained <- tree(richness ~ latitude, data=forest_ants)
plot(tree_trained, type="uniform")
text(tree_trained, pretty=0, digits=2)

#' We see that the tree splits latitude twice, first at 42.575 then at 42.18 to
#' give three terminal nodes. The predicted richness is at the end of the
#' terminal nodes. To make predictions, the model algorithm simply follows the
#' branches of the tree evaluating decisions. We start at the top of the
#' decision tree and work down. For example, say we want to predict species
#' richness for latitude 42.4. At the first decision node, 42.4 is less than
#' 42.575, so we follow the left branch to another decision node. At this node,
#' we find that 42.4 is greater than 42.18, so we follow the right branch, to a
#' terminal node, which gives the prediction of 10 species.
#'
#' Plot predictions across a grid of latitude values and compare with the data:

grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
grid_data <- data.frame(latitude=grid_latitude)
preds <- cbind(grid_data, richness=predict(fit, newdata=grid_data))

forest_ants |> 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=latitude, y=richness)) +
    coord_cartesian(ylim=c(0,20))

#' We see that for a single predictor variable, a regression tree partitions the
#' predictor (x axis) into segments (three segments in this case). We are
#' modeling any patterns or nonlinearities in the data as a step function for
#' which we can control the resolution (the number of partitions) through the
#' parameters of the tree algorithm.
#' 

#' Now let's look at the algorithms. We'll look at the **training algorithm**
#' first. This algorithm is called binary recursive partitioning.
#' 

# ```
# # Binary recursive partitioning
# define build_tree (y, x) {
#     if stop = TRUE
#         calculate prediction (mean of y)
#     else
#         partition the data in two along x
#         left branch:  build_tree( y, x[<x_split] )
#         right branch: build_tree( y, x[>x_split] )
# }
# ```

#' The algorithm is recursive because build_tree is a function and it calls
#' itself until it finally stops when a stopping rule becomes true. One pass
#' through this function either establishes a terminal node with a prediction,
#' or a decision node and two branches. Here is an R implementation that prints
#' out the tree structure. It has two stopping rules: stop if the number of
#' points in a partition falls below n_min, or stop if the branching depth d,
#' exceeds d_max. To split the data, it calls another function (which I have
#' read in ahead of time using `source()` above) to determine the best split of
#' the data, using an approach that minimizes the SSQ of the two means. We'll
#' look at that in a moment.

# Train a decision tree by recursive binary partitioning
# df:     data frame containing columns y, x
# n_min:  stopping rule, minimum number of points in a partition
# d_max:  stopping rule, maximum tree depth
# d:      current tree depth, initialized to 1
# node:   current node, initialized to 1
#
# output: prints out the tree structure; no return object

build_tree <- function( df, n_min, d_max, d=1, node=1 ) {
#   if stop = TRUE
    if ( nrow(df) < n_min | d >= d_max ) {
        # calculate and print prediction
        y_pred <- mean(df$y)
        print(paste(node, "leaf", round(y_pred, 2), sep="_"))
    } else {
        #find the best split of the data
        x_split <- best_split(df, n_min)
        #print decision node information
        print(paste(node, paste0("<", round(x_split, 2)), sep="_"))
    #   build left branch
        build_tree(df[df$x < x_split,], n_min, d_max, d + 1, 2 * node)
    #   build right branch
        build_tree(df[df$x >= x_split,], n_min, d_max, d + 1, 2 * node + 1)
    }
}

#' Now using this function with some generated data
set.seed(783)
x <- runif(100, -5, 5)
y <- rnorm(100, mean=100 + x - x ^ 2, sd=10)
x <- x + 5
df <- data.frame(x, y)
build_tree(df, n_min=5, d_max=3)

#' The printout shows a line for each node, identified by an integer index, and
#' the decision criteria, or the prediction if the node is a leaf. Thus node 1
#' says if x < 3.01, take the left branch. The left child branch is
#' node_left_child = 2 * node_parent, while the right child branch is
#' node_left_child = 2 * node_parent + 1. So, the left branch is node 2. We see
#' that this is again a decision node, with its left branch terminating at leaf
#' node 4 with a prediction of 79.17. There are 3 splits, or decision nodes, and
#' four leaf nodes, so the training algorithm has partitioned the data into 4
#' chunks.
#' 

#' Adding some book keeping to this algorithm, and another stopping rule that
#' ensures that we stop if there is no variation in x within a group.
#' 


# Array version of build tree with added check for zero variance
build_tree <- function( df, n_min, d_max, d=1, node=1 ) {

#   if stop = TRUE
    stop1 <- nrow(df) < 2 * n_min
    stop2 <- d >= d_max
    stop3 <- var(df$x) == 0 #can't split if no variance
    if ( stop1 | stop2 | stop3 ) {
    #   calculate and record prediction
        y_pred <- mean(df$y)
        result <- data.frame(node=node, type="leaf", split=NA, y_pred=y_pred)
    } else {
    #   find the best split of the data
        x_split <- best_split(df, n_min)
    #   record decision node and child branch information
        this_node <- data.frame(node=node, type="split", split=x_split, y_pred=NA)
    #   build left branch
        L_branch <- build_tree(df[df$x < x_split,], n_min, d_max, d + 1, 2 * node)
    #   build right branch
        R_branch <- build_tree(df[df$x >= x_split,], n_min, d_max, d + 1, 2 * node + 1)
        result <- rbind(this_node, L_branch, R_branch)
    }
    return(result)
}

#' To use this with the ants data, we'll rename columns as the build_tree
#' function requires the names "x" and "y".
train_tree <- function( df, n_min, d_max=5 ) {
    df <- df |> rename(x=latitude, y=richness)
    build_tree( df, n_min, d_max )
}

tree_trained <- train_tree(forest_ants, n_min=4, d_max=3)
plot_tree(tree_trained)


#' Code for the **model algorithm** to make predictions. The model consists of
#' the tree representation, along with this algorithm to read off the decisions.

# Makes predictions for a single x value from a tree that is in row index form
#
# t: tree in row index form (data frame)
# x: value of x to predict for (scalar)

predict_tree_x <- function( t, x ) {
    node <- 1
    while ( t$type[node] == "split" ) {
        if ( x < t$split[node] ) {
        #   Take left branch
            node <- 2 * node
        } else {
        #   Take right branch
            node <- 2 * node + 1
        }
    }
    return(t$y_pred[node])
}

tree_trained_ri <- to_tree_array(tree_trained)


# Now add loop to work with a vector of x values
predict_tree <- function( tree, x_new ) {
    t <- to_tree_array(tree)
    nx <- length(x_new)
    y_pred <- rep(NA, nx)
    for ( i in 1:nx ) {
        y_pred[i] <- predict_tree_x(t, x_new[i])
    }
    return(y_pred)
}


#' Plot predictions with the data.

grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
preds <- cbind(grid_data, richness=predict_tree(fit, x_new=grid_latitude))

forest_ants |> 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=latitude, y=richness)) +
    coord_cartesian(ylim=c(0,20))




#' In contrast, the following fit is for a deeper tree. We have modified the
#' stopping rules of the training algorithm to allow splits all the way to
#' individual data points.

tree_trained <- train_tree(forest_ants, n_min=1, d_max=8)
plot_tree(tree_trained)



#' Plot predictions with the data.

#' Plot predictions with the data.

grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
preds <- cbind(grid_data, richness=predict_tree(fit, x_new=grid_latitude))

forest_ants |> 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=latitude, y=richness)) +
    coord_cartesian(ylim=c(0,20))


#' For this tree the predictions follow the data except for the one case where
#' two data points shared the same latitude.
#' 

#' Now that we've examined the algorithms in some detail, we'll switch back to
#' using the `tree` package. The algorithms above would need more added to
#' handle multiple variables, categorical x variables, and the classification
#' case. There is more data in the ants dataset, including two more predictor
#' variables: habitat (bog or forest) and elevation (m).

ants <- read.csv("data/ants.csv") |> 
    select(-site) |> 
    mutate_if(is.character, factor)
head(ants)

#' Fit a tree that includes both latitude and habitat as predictors

fit <- tree(richness ~ latitude + habitat, data=ants)
plot(fit, type="uniform")
text(fit, pretty=0, digits=2)

#' We see the tree has nodes that split at both predictor variables. First it
#' splits by latitude, then it splits by habitat, then it splits by latitude
#' again. At the habitat nodes, bog is to the left while forest is to the right.
#' 

#' Plot the prediction from the fitted model

grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")))
preds <- cbind(grid_data, richness=predict(fit, newdata=grid_data))

ants |> 
    ggplot() +
    geom_point(aes(x=latitude, y=richness, col=habitat)) +
    geom_line(data=preds, aes(x=latitude, y=richness, col=habitat)) +
    coord_cartesian(ylim=c(0,20))

#' Plotting the predicted richness reveals that we have effectively modeled
#' richness as a nonlinear combination, or "interaction", of habitat and
#' latitude. The stepwise functions broadly (arguably crudely) capture the
#' pattern of a different nonlinear relationship between richness and latitude
#' in each habitat.
#' 

#' Now fit a tree with all three predictor variables

fit <- tree(richness ~ latitude + habitat + elevation, data=ants)
plot(fit, type="uniform")
text(fit, pretty=0, digits=2)

#' This tree has splits for all three predictor variables. First it splits by
#' latitude, then by habitat, and then it splits again by elevation in forest
#' habitat only. One interpretation is that elevation is important to predict
#' richness only in forest.
#'

#' It's harder to visualize the prediction from this fit since we have multiple
#' predictor dimensions. Here is one visualizaton:
#'

grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")),
    elevation=seq(min(ants$elevation), max(ants$elevation), length.out=51))
preds <- cbind(grid_data, richness=predict(fit, newdata=grid_data))

ants |> 
    ggplot() +
    geom_line(data=preds, 
              aes(x=latitude, y=richness, col=elevation, group=factor(elevation)),
              linetype=2) +
    geom_point(aes(x=latitude, y=richness, col=elevation)) +
    facet_wrap(vars(habitat)) +
    scale_color_viridis_c() +
    theme_bw()

#' The plot shows that elevation doesn't contribute to prediction in the bog
#' habitat but we're modeling different nonlinearities in the forest habitat for
#' lower and higher elevations. Effectively we are crudely modeling the
#' interaction between habitat, latitude, and elevation by dividing the
#' predictor space into 7 chunks. Since our goal is prediction we'd be most
#' interested in plotting predictions within some region to make a map of
#' predicted species richness for the area from which the data came. For that
#' we'd need maps of the predictor variables but such a visualization scales to
#' any number of predictor variables.
#' 

#' The prediction error (i.e. the out-of-sample error) from a regression tree
#' can be estimated by k-fold cross validation in the usual way.

# Function to divide a data set into random partitions for cross-validation
# n:       length of dataset (scalar, integer)
# k:       number of partitions (scalar, integer)
# return:  partition labels (vector, integer)
# 
random_partitions <- function(n, k) {
    min_n <- floor(n / k)
    extras <- n - k * min_n
    labels <- c(rep(1:k, each=min_n),rep(seq_len(extras)))
    partitions <- sample(labels, n)
    return(partitions)
}

# Function to perform k-fold CV for the tree model on ants data
# k:       number of partitions (scalar, integer)
# return:  CV error as MSE (scalar, numeric)
#
cv_tree_ants <- function(k) {
    ants$partition <- random_partitions(nrow(ants), k)
    e <- rep(NA, k)
    for ( i in 1:k ) {
        test_data <- subset(ants, partition == i)
        train_data <- subset(ants, partition != i)
        train_tree <- tree(richness ~ latitude + habitat + elevation, data=train_data)
        pred_richness <- predict(train_tree, newdata=test_data)
        e[i] <- mean((test_data$richness - pred_richness) ^ 2)
    }
    cv_error <- mean(e)
    return(cv_error)
}

#' Test the function

cv_tree_ants(k=5)
cv_tree_ants(k=nrow(ants)) #LOOCV

#' Running the above two lines of code multiple times we find lots of
#' variability in the prediction error estimate for 5-fold CV due to the
#' randomness of the partitions. LOOCV does not change because the tree model
#' and training algorithms are deterministic. As before, we'll need repeated
#' partitions for a more stable estimate of the 5-fold CV:

#+ cache=TRUE
set.seed(3127)
reps <- 500
cv_error <- rep(NA, reps)
for ( i in 1:reps ) {
    cv_error[i] <- cv_tree_ants(k=5)
}

#' A histogram suggests the CV replicates are well behaved
hist(cv_error)

#' Estimated error and its Monte Carlo error (about +/- 0.1)
mean(cv_error)
sd(cv_error) / sqrt(reps)

