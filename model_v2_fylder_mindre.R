pacman::p_load(
  tidyverse, tidymodels, themis, ranger, glmnet,
  kernlab, xgboost, rpart, future
)

model_data <- readRDS("data/model_data.rds")

##################################################
# 🔹 MODEL SPECS
##################################################

model_specs <- list(
  decision_tree = decision_tree(
    tree_depth = tune(), min_n = tune(), cost_complexity = tune()
  ) %>% set_engine("rpart") %>% set_mode("classification"),
  
  logistic_reg = logistic_reg(
    penalty = tune(), mixture = tune()
  ) %>% set_engine("glmnet") %>% set_mode("classification"),
  
  rand_forest = rand_forest(
    mtry = tune(), min_n = tune(), trees = 500
  ) %>% set_engine("ranger", importance = "permutation") %>% set_mode("classification"),
  
  svm_rbf = svm_rbf(
    cost = tune(), rbf_sigma = tune()
  ) %>% set_engine("kernlab") %>% set_mode("classification"),
  
  xgboost = boost_tree(
    trees = tune(), tree_depth = tune(),
    learn_rate = tune(), mtry = tune()
  ) %>% set_engine("xgboost") %>% set_mode("classification")
)

##################################################
# 🔹 FIXED PIPELINE FUNCTION
##################################################

run_model <- function(data, target, strata_var, recipe_steps) {
  
  set.seed(8)
  
  split <- initial_split(data, prop = 0.8, strata = !!rlang::sym(strata_var))
  train <- training(split)
  
  folds <- vfold_cv(train, v = 5, strata = !!rlang::sym(strata_var))
  
  # ✅ FIX: convert target safely to formula string
  fml <- as.formula(paste(target, "~ ."))
  rec <- recipe(fml, data = train)
  
  rec <- recipe_steps(rec)
  
  wf_set <- workflow_set(
    preproc = list(rec = rec),
    models = model_specs
  )
  
  metrics <- metric_set(roc_auc, accuracy, f_meas, sens, spec)
  
  plan(multisession)
  
  results <- wf_set %>%
    workflow_map(
      "tune_grid",
      resamples = folds,
      grid = 5,
      metrics = metrics,
      control = control_grid(save_pred = TRUE)
    )
  
  plan(sequential)
  
  best_id <- results %>%
    rank_results(select_best = TRUE) %>%
    filter(.metric == "f_meas") %>%
    slice_max(mean, n = 1) %>%
    pull(wflow_id)
  
  best_params <- results %>%
    extract_workflow_set_result(best_id) %>%
    select_best(metric = "f_meas")
  
  final_wf <- results %>%
    extract_workflow(best_id) %>%
    finalize_workflow(best_params)
  
  final_fit <- final_wf %>%
    last_fit(split, metrics = metrics)
  
  list(
    metrics = collect_metrics(final_fit),
    preds   = collect_predictions(final_fit),
    model   = final_wf
  )
}

##################################################
# 🔹 DATA PREP
##################################################

model_data_clean <- model_data %>%
  select(
    -pseudo_id,
    -subscription_cancel_date, -type, -reason,
    -expiration_date, -order_trackertag,
    -first_campaign_day, -last_campaign_day,
    -continued_subscription
  ) %>%
  mutate(churn = factor(churn, levels = c("0", "1")))

##################################################
# 🔹 MODEL 1: CHURN
##################################################

churn_results <- run_model(
  data = model_data_clean,
  target = "churn",   # ✅ MUST BE STRING NOW
  strata_var = "churn",
  recipe_steps = function(rec) {
    rec %>%
      step_rm(early_churn) %>%
      step_novel(all_nominal_predictors()) %>%
      step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
      step_zv(all_predictors()) %>%
      step_normalize(all_numeric_predictors()) %>%
      step_downsample(churn)
  }
)

churn_results$metrics

churn_results$preds %>%
  conf_mat(truth = churn, estimate = .pred_class)

churn_results$preds %>%
  roc_curve(truth = churn, .pred_1) %>%
  autoplot()

##################################################
# 🔹 MODEL 2: EARLY CHURN
##################################################

early_data <- model_data_clean %>%
  filter(churn == "0") %>%
  mutate(early_churn = factor(early_churn, levels = c("0", "1")))

early_results <- run_model(
  data = early_data,
  target = "early_churn",   # ✅ STRING
  strata_var = "early_churn",
  recipe_steps = function(rec) {
    rec %>%
      step_rm(churn) %>%
      step_novel(all_nominal_predictors()) %>%
      step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
      step_zv(all_predictors()) %>%
      step_normalize(all_numeric_predictors())
  }
)

early_results$metrics

early_results$preds %>%
  conf_mat(truth = early_churn, estimate = .pred_class)