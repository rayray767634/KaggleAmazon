library(vroom)
library(tidyverse)
library(DataExplorer)
library(dplyr)
library(GGally)
library(patchwork)
library(tidymodels)
library(lubridate)
library(poissonreg)
library(glmnet)
library(stacks)
library(dbarts)
library(sparklyr)
library(embed)



# read in data
amaz.test <- vroom("amazontest.csv")

amaz.train <- vroom("amazontrain.csv") %>%
  mutate_at('ACTION', as.factor)


# recipe
my_recipe <- recipe(ACTION ~ ., data = amaz.train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = amaz.train)

# EDA
table(baked$ACTION)
ggplot(data = baked, mapping = aes(x = ACTION)) +
  geom_bar()


# logistic regression
my_mod <- logistic_reg() %>% # type of model
  set_engine("glm")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = amaz.train) # fit the workflow

amazonlog_predictions <- predict(amazon_workflow,
                              new_data = amaz.test,
                              type = "prob") %>%
  mutate(ACTION = ifelse(.pred_1>.5,1,0)) %>%
  bind_cols(., amaz.test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep id and pred_1
  rename(Action=.pred_1) #rename pred1 to Action (for submission to Kaggle)
  


vroom_write(x=amazonlog_predictions, file="./AmazonLogPreds.csv", delim=",")


