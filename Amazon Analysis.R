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

amaz.train <- vroom("amazontrain.csv")

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




