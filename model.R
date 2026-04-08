pacman::p_load(
  tidyverse, tidymodels, themis, ranger, glmnet,
  kernlab, kknn, xgboost, naivebayes, rpart, future
)

# Indlæs data
model_data <- readRDS("data/model_data.rds")

glimpse(model_data)

# Rens data (fjern leakage)
model_data_clean <- model_data %>%
  select(
    -subscription_cancel_date,
    -type,
    -reason,
    -expiration_date,
    -order_trackertag,
    -first_campaign_day,
    -last_campaign_day
  ) %>%
  mutate(churn = factor(churn, levels = c("0", "1")))

# Opdel i trænings- og testdata
set.seed(8)

split <- initial_split(model_data_clean, prop = 0.8, strata = churn)
train_data <- training(split)
test_data  <- testing(split)

# Resampling (krydsvalidering)
folds <- vfold_cv(train_data, v = 5, strata = churn)

# Recipe (feature preprocessing)
rec <- recipe(churn ~ ., data = train_data) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_downsample(churn)

# Modeller (tuning specifikationer)

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

# Workflow-sæt
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

# Evalueringsmetrikker
metrics <- metric_set(roc_auc, accuracy, f_meas, sens, spec)

# Parallel opsætning
plan(multisession)

# Hyperparameter tuning
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

# Sammenlign modeller
results %>%
  rank_results(select_best = TRUE) %>%
  select(wflow_id, .metric, mean) %>%
  pivot_wider(names_from = .metric, values_from = mean) %>%
  arrange(-f_meas)

# Vælg bedste model (baseret på F1-score)
best_model_id <- results %>%
  rank_results(select_best = TRUE) %>%
  filter(.metric == "f_meas") %>%
  slice_max(mean, n = 1) %>%
  pull(wflow_id)

best_tuned <- results %>%
  extract_workflow_set_result(best_model_id) %>%
  select_best(metric = "f_meas")

# Endeligt workflow
final_wf <- results %>%
  extract_workflow(best_model_id) %>%
  finalize_workflow(best_tuned)

# Endelig modeltræning med last_fit
final_fit <- final_wf %>%
  last_fit(split, metrics = metrics)

# Metrikker på testdatasæt
collect_metrics(final_fit)

# Forudsigelser
test_preds <- collect_predictions(final_fit)

# Confusion matrix
test_preds %>%
  conf_mat(truth = churn, estimate = .pred_class)

# ROC-kurve
test_preds %>%
  roc_curve(truth = churn, .pred_1) %>%
  autoplot()

# Endelig model trænet på hele datasættet (valgfrit)
final_model <- fit(final_wf, model_data_clean)

final_model

show_notes(.Last.tune.result)