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
## Build Model
```
knn_model1 <- nearest_neighbor(neighbors = 7, weight_func = "optimal") %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_model2 <- nearest_neighbor(neighbors = 10, weight_func = "optimal") %>%
  set_mode("classification") %>%
  set_engine("kknn")

tree_model <- decision_tree(tree_depth = 10, min_n = 3) %>%
  set_mode("classification") %>%
  set_engine("rpart")
```
## Define Workflow & Fit
```
knn_workflow1 <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model1) %>%
  fit(train)

knn_workflow2 <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model2) %>%
  fit(train)

tree_workflow <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(tree_model)

tree_workflow_fit <- tree_workflow %>% fit(train)
tree_workflow_fit
```
## Score Model
```
knn_scored_train1 <- predict(knn_workflow1, train, type = "prob") %>%
  bind_cols(predict(knn_workflow1, train, type = "class")) %>%
  bind_cols(., train)

knn_scored_test1 <- predict(knn_workflow1, test, type = "prob") %>%
  bind_cols(predict(knn_workflow1, test, type = "class")) %>%
  bind_cols(., test)

knn_scored_train2 <- predict(knn_workflow2, train, type = "prob") %>%
  bind_cols(predict(knn_workflow2, train, type = "class")) %>%
  bind_cols(., train)

knn_scored_test2 <- predict(knn_workflow2, test, type = "prob") %>%
  bind_cols(predict(knn_workflow2, test, type = "class")) %>%
  bind_cols(., test)

tree_scored_train <- predict(tree_workflow_fit, train, type = "prob") %>%
  bind_cols(predict(tree_workflow_fit, train, type = "class")) %>%
  bind_cols(., train)

tree_scored_test <- predict(tree_workflow_fit, test, type = "prob") %>%
  bind_cols(predict(tree_workflow_fit, test, type = "class")) %>%
  bind_cols(., test)
```
## Evaluation Metrics
```
options(yardstick.event_first = FALSE)

knn_scored_train1 %>%
  metrics(churn, .pred_1, estimate = .pred_class) %>%
  mutate(part = "knn training") %>%
  bind_rows(knn_scored_test1 %>%
              metrics(churn, .pred_1, estimate = .pred_class) %>%
              mutate(part = "knn testing")) %>%
  filter(.metric %in% c('accuracy', 'roc_auc')) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

knn_scored_train2 %>%
  metrics(churn, .pred_1, estimate = .pred_class) %>%
  mutate(part = "knn training") %>%
  bind_rows(knn_scored_test2 %>%
              metrics(churn, .pred_1, estimate = .pred_class) %>%
              mutate(part = "knn testing")) %>%
  filter(.metric %in% c('accuracy', 'roc_auc')) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

tree_scored_train %>%
  metrics(churn, .pred_1, estimate = .pred_class) %>%
  mutate(part = "tree training") %>%
  bind_rows(tree_scored_test %>%
              metrics(churn, .pred_1, estimate = .pred_class) %>%
              mutate(part = "tree testing")) %>%
  filter(.metric %in% c('accuracy', 'roc_auc')) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
```
## Precision & Recall
### KNN 1
```
knn_scored_train1 %>%
  precision(churn, .pred_class) %>%
  mutate(part = "knn training") %>%
  bind_rows(knn_scored_test1 %>%
              precision(churn, .pred_class) %>%
              mutate(part = "knn testing"))

knn_scored_train1 %>%
  recall(churn, .pred_class) %>%
  mutate(part = "training") %>%
  bind_rows(knn_scored_test1 %>%
              recall(churn, .pred_class) %>%
              mutate(part = "testing"))
```
### KNN 2
```
knn_scored_train2 %>%
  precision(churn, .pred_class) %>%
  mutate(part = "knn training") %>%
  bind_rows(knn_scored_test2 %>%
              precision(churn, .pred_class) %>%
              mutate(part = "knn testing"))

knn_scored_train2 %>%
  recall(churn, .pred_class) %>%
  mutate(part = "training") %>%
  bind_rows(knn_scored_test2 %>%
              recall(churn, .pred_class) %>%
     mutate(part = "testing"))
```
![Snipaste_2023-05-25_18-24-59](https://github.com/dingy21/customer-churn-knn/assets/134649288/0be590d5-d3c5-411e-9ffc-8e2ba11d6a98)
## Confusion Matrix
### KNN 1
```
knn_scored_train1 %>%
  conf_mat(truth = churn, estimate = .pred_class, dnn = c("Prediction", "Truth")) %>%
  autoplot(type = "heatmap") +
  labs(title = "KNN 1 Training Confusion Matrix")

knn_scored_test1 %>%
  conf_mat(truth = churn, estimate = .pred_class, dnn = c("Prediction", "Truth")) %>%
  autoplot(type = "heatmap") +
  labs(title = "KNN 1 Testing Confusion Matrix")
```
<img width="252" alt="Picture1" src="https://github.com/dingy21/customer-churn-knn/assets/134649288/788b5ca6-79d6-4731-b322-fe3cf8ced2fe"><img width="252" alt="Picture2" src="https://github.com/dingy21/customer-churn-knn/assets/134649288/6cf22031-7a2c-486a-a243-524436d4e36d">
### KNN 2
```
knn_scored_train2 %>%
  conf_mat(truth = churn, estimate = .pred_class, dnn = c("Prediction", "Truth")) %>%
  autoplot(type = "heatmap") +
  labs(title = "KNN 2 Training Confusion Matrix")

knn_scored_test2 %>%
  conf_mat(truth = churn, estimate = .pred_class, dnn = c("Prediction", "Truth")) %>%
  autoplot(type = "heatmap") +
  labs(title = "KNN 2 Testing Confusion Matrix")
```
<img width="252" alt="Picture3" src="https://github.com/dingy21/customer-churn-knn/assets/134649288/372037e0-e1f1-486c-a1fc-c41f1cedc580">
<img width="252" alt="Picture4" src="https://github.com/dingy21/customer-churn-knn/assets/134649288/ff6c4eb8-0b81-4c98-a8f3-c3c69cbd21c6">
### Decision Tree
```
tree_scored_train %>%
  conf_mat(truth = churn, estimate = .pred_class, dnn = c("Prediction", "Truth")) %>%
  autoplot(type = "heatmap") +
  labs(title = "Training Confusion Matrix")

tree_scored_test %>%
  conf_mat(truth = churn, estimate = .pred_class, dnn = c("Prediction", "Truth")) %>%
  autoplot(type = "heatmap") +
  labs(title = "Training Confusion Matrix")
```
