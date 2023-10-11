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

# penalized logistic regression
my_recipe_pen <- recipe(ACTION ~ ., data = amaz.train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))


my_mod_pen <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

amazon_workflow_pen <- workflow() %>%
  add_recipe(my_recipe_pen) %>%
  add_model(my_mod_pen)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

# split data for CV

folds <- vfold_cv(amaz.train, v = 5, repeats = 1)

# run the cv
CV_results <- amazon_workflow_pen %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the workflow and fit it
final_wf_pen <- amazon_workflow_pen %>%
  finalize_workflow(bestTune) %>%
  fit(data = amaz.train)

# predict 
amazonlogpen_predictions <- predict(final_wf_pen,
                                 new_data = amaz.test,
                                 type = "prob") %>%
  mutate(ACTION = ifelse(.pred_1>.5,1,0)) %>%
  bind_cols(., amaz.test) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep id and pred_1
  rename(Action=.pred_1) #rename pred1 to Action (for submission to Kaggle)

vroom_write(x=amazonlogpen_predictions, file="./AmazonLogPenPreds.csv", delim=",")
