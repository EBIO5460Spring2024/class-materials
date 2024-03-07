#' ---
#' title: "Convolutional neural network"
#' author: Brett Melbourne
#' date: 07 Mar 2024
#' output:
#'     github_document
#' ---

#' To demonstrate transfer learning we'll continue to use the ecological subset
#' of the CIFAR100 data and compare a transfer learning approach to our
#' previous model trained from scratch.

#+ results=FALSE, message=FALSE, warning=FALSE
reticulate::use_condaenv(condaenv = "r-tensorflow")
library(ggplot2)
library(dplyr)
library(keras)
source("source/prep_cifar56eco.R")
tensorflow::set_random_seed(2726) #sets for tensorflow, keras, and R

#' In the previous script we prepared the CIFAR56eco dataset. Load that back in.
load("data_large/cifar56eco.RData")

#' Data preparation (as before)
x_train <- x_train / 255
x_test <- x_test / 255
y_train_int <- y_train #keep a copy of the integer version for labelling later
y_train <- to_categorical(y_train, 56)

#' Resize images for use with vgg16 model (224 x 224 x 3). We'll do this by
#' padding.

x_train_pad <- keras3::op_image_pad(x_train, top_padding=100, left_padding=100,
                            target_height=224, target_width=224)
?image_smart_resize

#' Load vgg16 pretrained model

vgg16 <- application_vgg16(weights="imagenet")
vgg16

#' Plot a random selection of predictions from the pretrained model.

selection <- sort(sample(1:dim(x_test)[1], 16))
par(mar=c(0,0,0,0), mfrow=c(4,4))
for ( i in selection ) {
    pred <- as.numeric(predict(vgg16, x_test[i,,,,drop=FALSE]))
    plot(as.raster(x_test[i,,,]))
    text(0, 30, paste("prediction =", eco_labels$name[which.max(pred)]), col="red", pos=4)
    text(0, 28, paste("prob =", round(pred[which.max(pred)],2)), col="red", pos=4)
    text(0, 26, paste("actual =", eco_labels$name[y_test[i,]+1]), col="red", pos=4)
} 

#' Train the model for our target application by disconnecting the convolutional
#' base, which you think of as a feature extraction tool, and training a new
#' dense layer for our 56 ecological categories.

vgg16_base <- application_vgg16(weights="imagenet", include_top=FALSE)
vgg16_base

freeze_weights(vgg16_base)

inputs <- layer_input(shape=c(224,224,3))
outputs <- inputs |>    
    vgg16_base() |>
#   Flatten with dropout regularization
    layer_flatten() |>
    layer_dropout(rate=0.5) |>
#   Standard dense layer
    layer_dense(units=512) |>
    layer_dropout(rate=0.5) |>
    layer_activation_relu() |>
#   Output layer with softmax (56 categories to predict)    
    layer_dense(units=56) |> 
    layer_activation_softmax()

modtfr1 <- keras_model(inputs, outputs)


#' Our old model for comparison

modcnn1 <- keras_model_sequential(input_shape=c(32,32,3)) |>
#   1st convolution-pool layer sequence
    layer_conv_2d(filters=32, kernel_size=c(3,3), padding="same") |>
    layer_activation_relu() |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
#   2nd convolution-pool layer sequence    
    layer_conv_2d(filters=64, kernel_size=c(3,3), padding="same") |> 
    layer_activation_relu() |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
#   3rd convolution-pool layer sequence    
    layer_conv_2d(filters=128, kernel_size=c(3,3), padding="same") |> 
    layer_activation_relu() |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
#   4th convolution-pool layer sequence
    layer_conv_2d(filters=256, kernel_size=c(3,3), padding="same") |> 
    layer_activation_relu() |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
#   Flatten with dropout regularization
    layer_flatten() |>
    layer_dropout(rate=0.5) |>
#   Standard dense layer
    layer_dense(units=512) |>
    layer_activation_relu() |>
#   Output layer with softmax (56 categories to predict)    
    layer_dense(units=56) |> 
    layer_activation_softmax()

#' Check the architecture

modcnn1

#' We see that the model has almost 1 million parameters! For example, in the
#' first convolutional layer we have 32 filters, each 3x3, for each of the 3
#' input channels (RGB), so 32 x 3 x 3 x 3 = 864 weights to which we add 32 bias
#' parameters (one for each output channel) to give 896 parameters. In the
#' second convolutional layer we have 64 x 3 x 3 x 32 + 64 = 18496, and so on.
#' At the input to the dense feedforward network where the array is flattened we
#' have 1024 nodes connected to 512 nodes, so 1024 x 512 weights + 512 biases =
#' 524800 parameters. Nevertheless, we do have a lot of data, about 86 million
#' pixels (28000 x 32 x 32 x 3).
#' 

#' Compile the model, specifying a `categorical_crossentropy` loss function,
#' which will be used in the gradient descent algorithm. This is a measure of
#' fit and accuracy on a likelihood scale. `RMSprop` is the default training
#' algorithm, a variant of stochastic gradient descent. We'll also collect a
#' second and more direct measure of accuracy.
#+ eval=FALSE

compile(modcnn1, loss="categorical_crossentropy", optimizer="rmsprop",
        metrics="accuracy")

#' Train the model using an 80/20 train/validate split to monitor progress. This
#' will take about 15 minutes on CPU or about 20 seconds on a single NVidia A100
#' GPU (e.g. on a CU Alpine compute node).
#+ eval=FALSE

fit(modcnn1, x_train, y_train, epochs=30, batch_size=128, 
    validation_split=0.2) -> history

#' Save the model (or load previously trained model)

# save_model_tf(modcnn1, "08_3_convolutional_nnet_files/saved/modcnn1")
# save(history, file="08_3_convolutional_nnet_files/saved/modcnn1_history.Rdata")
modcnn1 <- load_model_tf("08_3_convolutional_nnet_files/saved/modcnn1")
load("08_3_convolutional_nnet_files/saved/modcnn1_history.Rdata")

#' Plotting the training history, we see evidence of overfitting after only 1 or
#' two epochs as the validation loss climbs. While the training accuracy
#' improves, the validation accuracy is stuck at about 40%. This is obviously
#' not impressive!

plot(history, smooth=FALSE)

#' Plot a random selection of predictions. While the model is incorrect on many
#' images, it is remarkable that it predicts many correctly (much better than
#' random guessing) and those that it gets wrong, you can often see how the
#' image resembles the model's prediction.

selection <- sort(sample(1:dim(x_test)[1], 16))
par(mar=c(0,0,0,0), mfrow=c(4,4))
for ( i in selection ) {
    pred <- as.numeric(predict(modcnn1, x_test[i,,,,drop=FALSE]))
    plot(as.raster(x_test[i,,,]))
    text(0, 30, paste("prediction =", eco_labels$name[which.max(pred)]), col="red", pos=4)
    text(0, 28, paste("prob =", round(pred[which.max(pred)],2)), col="red", pos=4)
    text(0, 26, paste("actual =", eco_labels$name[y_test[i,]+1]), col="red", pos=4)
} 

#' Predictions and overall accuracy on the hold out test set (about 42%)

pred_prob <- predict(modcnn1, x_test)
pred_cat <- as.numeric(k_argmax(pred_prob))
mean(pred_cat == drop(y_test))

#' Plot probabilities for the same selection of test cases as above

nr <- nrow(pred_prob)
pred_prob |> 
    data.frame() |>
    mutate(case=seq(nr)) |>
    tidyr::pivot_longer(cols=starts_with("X"), names_to="category", values_to="probability") |> 
    mutate(category=as.integer(sub("X", "", category)) - 1) |> 
    filter(case %in% selection) |> 
    ggplot() +
    geom_point(aes(x=category, y=probability)) +
    facet_wrap(vars(case), nrow=4, ncol=4, labeller=label_both)

