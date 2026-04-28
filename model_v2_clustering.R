pacman::p_load(
  tidyverse, tidymodels, themis, ranger, glmnet,
  kernlab, kknn, xgboost, naivebayes, rpart, future, vip
)

# Indlæs data
model_data <- readRDS("data/model_data.rds")
cluster_mapping <- readRDS("data/cluster_mapping.rds")
coolbehavior <- readRDS("data/coolbehavior.rds")

glimpse(coolbehavior)

### dashboard data til power BI
  final_dashboard_data <- model_data %>%
    left_join(cluster_mapping, by = "pseudo_id")
  
  final_dashboard_data$age <- round(final_dashboard_data$age)
  
  merged_dashboard_data <- final_dashboard_data %>%
    left_join(coolbehavior, by = "pseudo_id")
  
  # Indsæt 0 i stedet for NA for de 70 "Spøgelses-brugere" 
  merged_dashboard_data <- merged_dashboard_data %>%
    mutate(across(
      c(n_visits, n_unique_pages, avg_scroll, starts_with("share_")),
      ~replace_na(.x, 0)
    ))
  
  write_csv(merged_dashboard_data, "data/dashboard_data.csv")

# Join cluster labels and clean initial data
model_data_clean <- model_data %>%
  # 1. Join the clusters
  left_join(cluster_mapping, by = "pseudo_id") %>%
  # 2. Convert label to factor for modeling
  mutate(cluster_label = as.factor(cluster_label)) %>%
  # 3. Rens data (fjern leakage og ID'er)
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

# Check that cluster_label is present
glimpse(model_data_clean)

# Split
set.seed(8)
split <- initial_split(model_data_clean, prop = 0.8, strata = churn)
train_data <- training(split)
test_data  <- testing(split)

# CV
folds <- vfold_cv(train_data, v = 5, strata = churn)

# Recipe
rec <- recipe(churn ~ ., data = train_data) %>%
  step_rm(early_churn) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(churn)
# step_upsample(churn)
# step_downsample(churn)

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


# Train
plan(multisession)

set.seed(8)

# We use suppressMessages to hide the "Fold X: model Y/Z" notes 
# and suppressWarnings to hide the Precision/Recall NA warnings.
results <- suppressMessages(suppressWarnings(
  workflow_set_obj %>%
    workflow_map(
      "tune_grid",
      resamples = folds,
      grid = 5,
      metrics = metrics,
      verbose = FALSE, # Switches off the tidymodels progress logger
      control = control_grid(
        save_pred = TRUE,
        save_workflow = TRUE
      )
    )
))

plan(sequential)

# 
# plan(multisession)
# 
# set.seed(8)
# results <- workflow_set_obj %>%
#   workflow_map(
#     "tune_grid",
#     resamples = folds,
#     grid = 5,
#     metrics = metrics,
#     control = control_grid(
#       verbose = TRUE,
#       save_pred = TRUE,
#       save_workflow = TRUE
#     )
#   )
# 
# plan(sequential)
# 

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

best_model_id

final_fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 20) +
  theme_minimal() +
  labs(title = "What Drives Churn?",
       subtitle = "Variables ranked by Importance (Random Forest)")

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

best_model_id_early

final_fit_early %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 20) +
  theme_minimal() +
  labs(title = "What Drives Early Churn?",
       subtitle = "Variables ranked by Importance (Decision Tree)")




##################################################
# 📊 RESULTS VISUALIZATION & COMPARISON
##################################################

# 1. Show the "Leaderboard" for Model 1 (Churn)
# This lists all models ranked by F-Measure
cat("\n--- Model 1: Churn Leaderboard ---\n")
results %>%
  rank_results(rank_metric = "f_meas", select_best = TRUE) %>%
  select(wflow_id, .metric, mean, std_err, rank) %>%
  filter(.metric == "f_meas") %>%
  print()

# 2. Plot Model 1 Comparison
# This gives you a visual look at how the different engines compared
p1 <- autoplot(results, metric = "f_meas", select_best = TRUE) +
  geom_text(aes(label = wflow_id), vjust = -1, size = 3) +
  labs(title = "Churn Model Comparison (F-Meas)",
       subtitle = "Best version of each algorithm") +
  theme_minimal()

print(p1)

# 3. Show the "Leaderboard" for Model 2 (Early Churn)
cat("\n--- Model 2: Early Churn Leaderboard ---\n")
results_early %>%
  rank_results(rank_metric = "f_meas", select_best = TRUE) %>%
  select(wflow_id, .metric, mean, std_err, rank) %>%
  filter(.metric == "f_meas") %>%
  print()

# 4. Plot Model 2 Comparison
p2 <- autoplot(results_early, metric = "f_meas", select_best = TRUE) +
  geom_text(aes(label = wflow_id), vjust = -1, size = 3) +
  labs(title = "Early Churn Model Comparison (F-Meas)",
       subtitle = "Best version of each algorithm") +
  theme_minimal()

print(p2)

# 5. Identify exactly which Hyperparameters the winner used
cat("\n--- Best Model Hyperparameters ---\n")
cat("Churn Model:", best_model_id, "\n")
print(best_tuned)

cat("\nEarly Churn Model:", best_model_id_early, "\n")
print(best_tuned_early)





# 
# # 1. Get predictions for all users
# final_results <- final_fit %>%
#   extract_workflow() %>%
#   augment(model_data_clean)
# 
# # 2. Create the Cluster-Risk Matrix
# risk_profile <- final_results %>%
#   group_by(cluster_label) %>%
#   summarise(
#     n_users = n(),
#     # Model's average predicted probability of churn
#     avg_predicted_risk = mean(.pred_1), 
#     # Actual churn recorded in data
#     actual_churn_rate = mean(churn == "1"),
#     # Early Churn rate within this cluster
#     early_churn_rate = mean(early_churn == 1),
#     # Loyalty score (How many stayed)
#     loyalty_rate = mean(churn == "0")
#   ) %>%
#   arrange(desc(avg_predicted_risk))
# 
# print(risk_profile)
