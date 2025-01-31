remotes::install_github("rstudio/tensorflow")
reticulate::install_python()
library(tensorflow)

install_tensorflow(envname = "r-tensorflow")
install.packages("keras")
library(keras)
install_keras()
library(tensorflow)

library(keras)
library(ggplot2)

train_dir <-  file.path("D:/JJZ/ML/Project/archive/Train_Test_Valid/Train")
validation_dir <-  file.path("D:/JJZ/ML/Project/archive/Train_Test_Valid/valid")

###### we start at a simple model

## data_generator

train_datagen = image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(224, 224),
  batch_size = 15,
  class_mode = 'sparse'
)

test_datagen <- image_data_generator(rescale = 1/255)

validation_generator <- flow_images_from_directory(
  directory = validation_dir,
  generator = test_datagen,
  target_size = c(224, 224),
  batch_size = 20,
  class_mode = 'sparse'
)

## model

model1 <- keras_model_sequential() %>%
  layer_conv_2d( filter = 16, kernel_size = c(3,3), padding = "same",
                 input_shape = c(224, 224, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d( filter = 32, kernel_size = c(3,3), padding = "same",
                 activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2))  %>%
  layer_flatten() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(6, activation = "softmax")


summary(model1)

model1 %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = "accuracy"
  )

history1 <- model1 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40, 
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 1 
)

plot(history1)

###### data agaumatation

train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
# same as model1
model2 <- keras_model_sequential() %>%
  layer_conv_2d( filter = 16, kernel_size = c(3,3), padding = "same",
                 input_shape = c(224, 224, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d( filter = 32, kernel_size = c(3,3), padding = "same",
                 activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2))  %>%
  layer_flatten() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(6, activation = "softmax")
model2 %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = "accuracy"
  )

history2 <- model2 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40, 
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 3 
)
plot(history2)

model3 <- keras_model_sequential() %>%
  layer_conv_2d( filter = 16, kernel_size = c(3,3), padding = "same",
                 input_shape = c(224, 224, 3), activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d( filter = 32, kernel_size = c(3,3), padding = "same",
                 activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(6, activation = "softmax")
summary(model3)
model3 %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = "accuracy"
  )

history3 <- model3 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40, 
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 3 
)
plot(history3)
# droup out

model3 %>% save_model_hdf5("model3_drop2.h5")

model33 <- keras_model_sequential() %>%
  layer_conv_2d( filter = 16, kernel_size = c(3,3), padding = "same",
                 input_shape = c(224, 224, 3), activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d( filter = 32, kernel_size = c(3,3), padding = "same",
                 activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(6, activation = "softmax")
summary(model33)
model33 %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = "accuracy"
  )

history33 <- model33 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40, 
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 3 
)
plot(history33)



model4 <- keras_model_sequential() %>%
  layer_conv_2d( filter = 32, kernel_size = c(3,3), padding = "same",
                 input_shape = c(224, 224, 3), activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2))  %>%
  layer_conv_2d( filter = 128, kernel_size = c(3,3), padding = "same",
                 activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d( filter = 32, kernel_size = c(3,3), padding = "same",
                 activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(6, activation = "softmax")
summary(model4)
model4 %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = "accuracy"
  )

history4 <- model4 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40, 
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 3 
)
plot(history4)


model55 <- keras_model_sequential() %>%
  layer_conv_2d( filter = 16, kernel_size = c(3,3), padding = "same",
                 input_shape = c(224, 224, 3), activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 32, kernel_size = c(4,4), activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d( filter = 32, kernel_size = c(4,4), padding = "same",
                 activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(6, activation = "softmax")
summary(model55)
model55 %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = "accuracy"
  )

history55 <- model55 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40, 
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 3 
)
plot(history55)



model5 <- keras_model_sequential() %>%
  layer_conv_2d( filter = 32, kernel_size = c(3,3), padding = "same",
                 input_shape = c(224, 224, 3), activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 64, kernel_size = c(4,4), activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2))  %>%
  layer_conv_2d( filter = 128, kernel_size = c(4,4), padding = "same",
                 activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d( filter = 64, kernel_size = c(3,3), padding = "same",
                 activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(6, activation = "softmax")
summary(model5)
model5 %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = "accuracy"
  )

history5 <- model5 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40, 
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 3 
)
plot(history5)




conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224, 224, 3)
)
summary(conv_base)

modelt <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 6, activation = "sigmoid")
summary(modelt)

freeze_weights(conv_base)

modelt %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.0005),
    metrics = "accuracy"
  )

historyt <- modelt %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40, 
  epochs = 20,
  validation_data = validation_generator,
  validation_steps = 3 
)
plot(historyt)



modelt1 <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 6, activation = "sigmoid")
freeze_weights(conv_base, to = "block5_conv1")

modelt1 %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.0005),
  metrics = c("accuracy")
)

historyt1 <- modelt1 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40,
  epochs = 20,
  validation_data = validation_generator,
  validation_steps = 3
)

model %>% save_model_hdf5("cats_and_dogs_filtered_augmented_drop.h5") 

plot(historyt2)
summary(modelt1)
plot(historyt1)







###unfreeze based on modelt

unfreeze_weights(conv_base, from = "block5_conv1")

modelt %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.0001),
  metrics = c("accuracy")
)

historyt2 <- modelt %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40,
  epochs = 20,
  validation_data = validation_generator,
  validation_steps = 3
)









modelt1 <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 6, activation = "sigmoid")
freeze_weights(conv_base, to = "block5_conv1")

modelt1 %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.0005),
  metrics = c("accuracy")
)

historyt1 <- modelt1 %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = 40,
  epochs = 20,
  validation_data = validation_generator,
  validation_steps = 3
)

model %>% save_model_hdf5("cats_and_dogs_filtered_augmented_drop.h5") 

plot(historyt2)
summary(modelt)
summary(modelt1)
summary(modelt2)

