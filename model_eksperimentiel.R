# --- 1. Load & Merge ---
pacman::p_load(tidyverse, tidymodels, themis, ranger, glmnet, vip)

customer_behavior <- readRDS("data/coolbehavior.rds") 
model_data <- readRDS("data/model_data.rds")

model_data_clean <- model_data %>%
  # Use inner_join to only model users who actually showed up
  inner_join(customer_behavior, by = "pseudo_id") %>%
  mutate(across(where(is.numeric), ~replace_na(.x, 0)))

# --- 2. Data Split ---
set.seed(8)
split <- initial_split(model_data_clean, prop = 0.8, strata = churn)
train_data <- training(split)
folds <- vfold_cv(train_data, v = 5, strata = churn)

# --- 3. The "Pure Behavior" Recipe ---
rec <- recipe(churn ~ ., data = train_data) %>%
  update_role(pseudo_id, new_role = "id") %>%
  
  # 1. THE AGGRESSIVE LEAKAGE FILTER
  step_rm(any_of(c(
    "early_churn", "account_active_days", "days_since_last_visit",
    "previous_subscriptions", "previous_trials", "continued_subscription",
    "subscription_status", "is_active" # Add any other status flags here
  ))) %>% 
  # Remove all clusters and newsletter info that might be proxies
  step_rm(contains("cluster"), contains("newsletter"), contains("after_order")) %>%
  step_rm(has_type("date")) %>% 
  
  # 2. Pre-processing
  step_mutate(eng_per_page = avg_scroll / (n_unique_pages + 1)) %>%
  step_log(n_unique_pages, n_visits, offset = 1) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(churn)

# --- 4. Model Specs (Simplified to Random Forest for speed) ---
rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger", importance = "permutation") %>% 
  set_mode("classification")

workflow_obj <- workflow() %>% add_recipe(rec) %>% add_model(rf_spec)

# --- 5. Tuning ---
set.seed(8)
results <- tune_grid(
  workflow_obj,
  resamples = folds,
  grid = 10,
  metrics = metric_set(roc_auc, accuracy)
)

# --- 6. Final Fit ---
best_tuned <- select_best(results, metric = "roc_auc")
final_fit <- finalize_workflow(workflow_obj, best_tuned) %>% last_fit(split)

# --- 7. Results ---
print("Metrics (Should no longer be 1.0):")
print(collect_metrics(final_fit))

print("New Feature Importance (The real drivers):")
final_fit %>% extract_fit_parsnip() %>% vip(num_features = 15)