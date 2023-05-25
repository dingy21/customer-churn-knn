# Customer Churn Prediction Project
Customer churn, also known as customer attrition, occurs when customers stop doing business with a company or stop using a company’s services. In this project, two KNN machine learning models(different values of K) were built to identify likely churners, so that the company can incentivize the churners to stay with the company.
## Load Library
```
library(tidyverse)
library(tidymodels)
library(janitor)
```
## Import Data
```
churn <- read_csv("Churn_training.csv") %>% clean_names()
churn_kaggle <- read_csv("Churn_holdout.csv") %>% clean_names()

head(churn)
churn %>% skimr::skim()
```
## Data Preparation
```
churn_prep <- churn %>%
  mutate(churn = as.factor(churn)) %>%
  mutate_if(is.character, factor)

churn_kaggle <- churn_kaggle %>%
  mutate_if(is.character, factor)
```
## Data Partition
```
set.seed(123)

x <- initial_split(churn_prep, prop = 0.7)
train <- training(x)
test <- testing(x)

sprintf("Train PCT : %1.2f%%", nrow(train)/nrow(churn) * 100)
sprintf("Test PCT : %1.2f%%", nrow(test)/nrow(churn) * 100)
```
## Define Recipe
```
knn_recipe <- recipe(churn ~ total_billed + number_phones + paperless_billing + streaming_minutes + streaming_plan + 
                     prev_balance + monthly_minutes + late_payments + payment_method, data = train) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())

bake(knn_recipe %>% prep(), train, composition = "tibble")
```
