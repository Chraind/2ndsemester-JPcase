pacman::p_load(
  tidyverse, tidymodels, themis, ranger, glmnet,
  kernlab, kknn, xgboost, naivebayes, rpart, future
)

# Indlæs data
model_data <- readRDS("data/model_data.rds")

glimpse(model_data)

##################################################
# 🔹 MODEL 1: CHURN (TASK 1 + 2)
##################################################

# Rens data (fjern leakage, MEN behold early_churn til senere)
model_data_clean <- model_data %>%
  select(
    -pseudo_id,
    -subscription_cancel_date,
    -type,
    -reason,
    -expiration_date,
    -order_trackertag,
    -first_campaign_day,
    -last_campaign_day,
    -continued_subscription
  ) %>%
  mutate(churn = factor(churn, levels = c("0", "1")))

# Split
set.seed(8)
split <- initial_split(model_data_clean, prop = 0.8, strata = churn)
train_data <- training(split)
test_data  <- testing(split)

# CV
folds <- vfold_cv(train_data, v = 5, strata = churn)

# Recipe
rec <- recipe(churn ~ ., data = train_data) %>%
  step_rm(early_churn) %>%   # VERY IMPORTANT: avoid leakage
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_downsample(churn)

# Models
decision_tree_spec <- decision_tree(
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

log_reg_spec <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

rand_forest_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

svm_rbf_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Workflow
workflow_set_obj <- workflow_set(
  preproc = list(rec = rec),
  models = list(
    decision_tree = decision_tree_spec,
    logistic_reg  = log_reg_spec,
    rand_forest   = rand_forest_spec,
    svm_rbf       = svm_rbf_spec,
    xgboost       = xgb_spec
  )
)

metrics <- metric_set(roc_auc, accuracy, f_meas, sens, spec)

plan(multisession)

set.seed(8)
results <- workflow_set_obj %>%
  workflow_map(
    "tune_grid",
    resamples = folds,
    grid = 5,
    metrics = metrics,
    control = control_grid(
      verbose = TRUE,
      save_pred = TRUE,
      save_workflow = TRUE
    )
  )

plan(sequential)

# Best model
best_model_id <- results %>%
  rank_results(select_best = TRUE) %>%
  filter(.metric == "f_meas") %>%
  slice_max(mean, n = 1) %>%
  pull(wflow_id)

best_tuned <- results %>%
  extract_workflow_set_result(best_model_id) %>%
  select_best(metric = "f_meas")

final_wf <- results %>%
  extract_workflow(best_model_id) %>%
  finalize_workflow(best_tuned)

# Final evaluation
final_fit <- final_wf %>%
  last_fit(split, metrics = metrics)

collect_metrics(final_fit)

test_preds <- collect_predictions(final_fit)

test_preds %>%
  conf_mat(truth = churn, estimate = .pred_class)

test_preds %>%
  roc_curve(truth = churn, .pred_1) %>%
  autoplot()

##################################################
# 🔹 MODEL 2: EARLY CHURN (TASK 3)
##################################################

# Only customers who continued
early_data <- model_data_clean %>%
  filter(churn == "0") %>%
  mutate(early_churn = factor(early_churn, levels = c("0", "1")))

# Split
set.seed(8)
split_early <- initial_split(early_data, prop = 0.8, strata = early_churn)
train_early <- training(split_early)
test_early  <- testing(split_early)

# CV
folds_early <- vfold_cv(train_early, v = 5, strata = early_churn)

# Recipe
rec_early <- recipe(early_churn ~ ., data = train_early) %>%
  step_rm(churn) %>%  # remove parent target
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())
#  step_downsample(early_churn)

# Workflow (reuse same models!)
workflow_set_early <- workflow_set(
  preproc = list(rec = rec_early),
  models = list(
    decision_tree = decision_tree_spec,
    logistic_reg  = log_reg_spec,
    rand_forest   = rand_forest_spec,
    svm_rbf       = svm_rbf_spec,
    xgboost       = xgb_spec
  )
)

# Train
plan(multisession)

set.seed(8)
results_early <- workflow_set_early %>%
  workflow_map(
    "tune_grid",
    resamples = folds_early,
    grid = 5,
    metrics = metrics
  )

plan(sequential)

# Best model
best_model_id_early <- results_early %>%
  rank_results(select_best = TRUE) %>%
  filter(.metric == "f_meas") %>%
  slice_max(mean, n = 1) %>%
  pull(wflow_id)

best_tuned_early <- results_early %>%
  extract_workflow_set_result(best_model_id_early) %>%
  select_best(metric = "f_meas")

final_wf_early <- results_early %>%
  extract_workflow(best_model_id_early) %>%
  finalize_workflow(best_tuned_early)

# Final evaluation
final_fit_early <- final_wf_early %>%
  last_fit(split_early, metrics = metrics)

collect_metrics(final_fit_early)

test_preds_early <- collect_predictions(final_fit_early)

test_preds_early %>%
  conf_mat(truth = early_churn, estimate = .pred_class)