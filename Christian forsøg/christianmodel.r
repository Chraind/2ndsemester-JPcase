pacman::p_load(
  tidyverse, tidymodels, themis, ranger, glmnet,
  kernlab, kknn, xgboost, naivebayes, rpart, future, vip
)

# ── Indlæs data ────────────────────────────────────────────────────────────────
model_data      <- readRDS("data/model_data.rds")
cluster_mapping <- readRDS("data/cluster_mapping.rds")
coolbehavior    <- readRDS("data/coolbehavior.rds")

# ── Dashboard data til Power BI ────────────────────────────────────────────────
final_dashboard_data <- model_data %>%
  left_join(cluster_mapping, by = "pseudo_id")

final_dashboard_data$age <- round(final_dashboard_data$age)

merged_dashboard_data <- final_dashboard_data %>%
  left_join(coolbehavior, by = "pseudo_id") %>%
  mutate(across(
    c(n_visits, n_unique_pages, avg_scroll, starts_with("share_")),
    ~ replace_na(.x, 0)
  ))

write_csv(merged_dashboard_data, "data/dashboard_data.csv")

# ── Klargør model_data_clean ───────────────────────────────────────────────────
model_data_clean <- model_data %>%
  left_join(cluster_mapping, by = "pseudo_id") %>%
  mutate(
    cluster_label = as.factor(cluster_label),
    churn         = factor(churn, levels = c("1", "0")),
    early_churn   = factor(early_churn, levels = c("0", "1"))
  ) %>%
  select(-pseudo_id, -continued_subscription)

glimpse(model_data_clean)

model_data_clean %>%
  count(churn) %>%
  mutate(prop = n / sum(n))

saveRDS(model_data_clean, "data/model_data_clean.rds")

# ── Model specs ────────────────────────────────────────────────────────────────
decision_tree_spec <- decision_tree(
  tree_depth = tune(), min_n = tune(), cost_complexity = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

log_reg_spec <- logistic_reg(
  penalty = tune(), mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

rand_forest_spec <- rand_forest(
  mtry = tune(), min_n = tune(), trees = 500
) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

svm_rbf_spec <- svm_rbf(
  cost = tune(), rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

xgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(),
  learn_rate = tune(), mtry = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

metrics <- metric_set(roc_auc, accuracy, f_meas, sens, spec)

# ── VIP hjælpefunktion ─────────────────────────────────────────────────────────
plot_vip <- function(fit, title) {
  tryCatch(
    {
      # Forsøg 1: standard vip (virker for RF, XGBoost, decision tree)
      fit %>%
        extract_fit_parsnip() %>%
        vip(num_features = 20) +
        theme_minimal() +
        labs(title = title)
    },
    error = function(e) {
      tryCatch(
        {
          # Forsøg 2: koefficienter (virker for logistisk regression)
          fit %>%
            extract_fit_parsnip() %>%
            tidy() %>%
            filter(term != "(Intercept)") %>%
            slice_max(abs(estimate), n = 20) %>%
            mutate(term = reorder(term, abs(estimate))) %>%
            ggplot(aes(x = abs(estimate), y = term, fill = estimate > 0)) +
            geom_col() +
            scale_fill_manual(
              values = c("TRUE" = "#2196F3", "FALSE" = "#F44336"),
              labels = c("TRUE" = "Reducerer churn", "FALSE" = "Øger churn")
            ) +
            labs(title = title, x = "Absolut koefficient", y = NULL, fill = NULL) +
            theme_minimal()
        },
        error = function(e2) {
          # Forsøg 3: SVM og andre modeller uden variable importance
          cat("⚠️  Variable importance ikke tilgængeligt for denne modeltype.\n")
          cat("   Bedste model:", title, "\n")
          ggplot() +
            annotate("text", x = 0.5, y = 0.5,
                     label = paste("Variable importance\nikke tilgængeligt for\ndenne modeltype"),
                     size = 6, hjust = 0.5) +
            theme_void() +
            labs(title = title)
        }
      )
    }
  )
}

##################################################
# MODEL 1: CHURN UNDER KAMPAGNEN
##################################################

# Split
set.seed(8)
split      <- initial_split(model_data_clean, prop = 0.8, strata = churn)
train_data <- training(split)
test_data  <- testing(split)

folds <- vfold_cv(train_data, v = 5, strata = churn)

# Recipe
rec <- recipe(churn ~ ., data = train_data) %>%
  step_rm(early_churn) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(churn)

# Workflow set
workflow_set_obj <- workflow_set(
  preproc = list(rec = rec),
  models  = list(
    decision_tree = decision_tree_spec,
    logistic_reg  = log_reg_spec,
    rand_forest   = rand_forest_spec,
    svm_rbf       = svm_rbf_spec,
    xgboost       = xgb_spec
  )
)

# Træn
plan(multisession)
set.seed(8)
results <- suppressMessages(suppressWarnings(
  workflow_set_obj %>%
    workflow_map(
      "tune_grid",
      resamples = folds,
      grid      = 20,
      metrics   = metrics,
      verbose   = FALSE,
      control   = control_grid(save_pred = TRUE, save_workflow = TRUE)
    )
))
plan(sequential)

# Leaderboard
cat("\n--- Model 1: Churn Leaderboard (roc_auc) ---\n")
results %>%
  rank_results(rank_metric = "roc_auc", select_best = TRUE) %>%
  filter(.metric == "roc_auc") %>%
  select(wflow_id, .metric, mean, std_err, rank) %>%
  print()

autoplot(results, metric = "roc_auc", select_best = TRUE) +
  labs(title = "Churn Model Comparison (ROC AUC)") +
  theme_minimal()

# Bedste model
best_model_id <- results %>%
  rank_results(select_best = TRUE) %>%
  filter(.metric == "roc_auc") %>%
  slice_max(mean, n = 1) %>%
  pull(wflow_id)

cat("\nBedste churn model:", best_model_id, "\n")

best_tuned <- results %>%
  extract_workflow_set_result(best_model_id) %>%
  select_best(metric = "roc_auc")

final_wf <- results %>%
  extract_workflow(best_model_id) %>%
  finalize_workflow(best_tuned)

final_fit <- final_wf %>%
  last_fit(split, metrics = metrics)

# Evaluer
cat("\n--- Model 1: Test metrics ---\n")
collect_metrics(final_fit) %>% print()

collect_predictions(final_fit) %>%
  conf_mat(truth = churn, estimate = .pred_class)

collect_predictions(final_fit) %>%
  roc_curve(truth = churn, .pred_1) %>%
  autoplot() +
  labs(title = "ROC Curve — Churn Model")

plot_vip(final_fit, "What Drives Churn?")

##################################################
# MODEL 2: EARLY CHURN
##################################################

# Kun kunder der fortsatte
early_data <- model_data_clean %>%
  filter(churn == "0")

cat("\nEarly churn fordeling:\n")
early_data %>%
  count(early_churn) %>%
  mutate(prop = n / sum(n)) %>%
  print()

# Split
set.seed(8)
split_early <- initial_split(early_data, prop = 0.8, strata = early_churn)
train_early <- training(split_early)
test_early  <- testing(split_early)

folds_early <- vfold_cv(train_early, v = 5, strata = early_churn)

# Recipe
rec_early <- recipe(early_churn ~ ., data = train_early) %>%
  step_rm(churn) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(early_churn)

# Workflow set
workflow_set_early <- workflow_set(
  preproc = list(rec = rec_early),
  models  = list(
    decision_tree = decision_tree_spec,
    logistic_reg  = log_reg_spec,
    rand_forest   = rand_forest_spec,
    svm_rbf       = svm_rbf_spec,
    xgboost       = xgb_spec
  )
)

# Træn
plan(multisession)
set.seed(8)
results_early <- suppressMessages(suppressWarnings(
  workflow_set_early %>%
    workflow_map(
      "tune_grid",
      resamples = folds_early,
      grid      = 20,
      metrics   = metrics,
      control   = control_grid(save_pred = TRUE, save_workflow = TRUE)
    )
))
plan(sequential)

# Leaderboard
cat("\n--- Model 2: Early Churn Leaderboard (roc_auc) ---\n")
results_early %>%
  rank_results(rank_metric = "roc_auc", select_best = TRUE) %>%
  filter(.metric == "roc_auc") %>%
  select(wflow_id, .metric, mean, std_err, rank) %>%
  print()

autoplot(results_early, metric = "roc_auc", select_best = TRUE) +
  labs(title = "Early Churn Model Comparison (ROC AUC)") +
  theme_minimal()

# Bedste model
best_model_id_early <- results_early %>%
  rank_results(select_best = TRUE) %>%
  filter(.metric == "roc_auc") %>%
  slice_max(mean, n = 1) %>%
  pull(wflow_id)

cat("\nBedste early churn model:", best_model_id_early, "\n")

best_tuned_early <- results_early %>%
  extract_workflow_set_result(best_model_id_early) %>%
  select_best(metric = "roc_auc")

final_wf_early <- results_early %>%
  extract_workflow(best_model_id_early) %>%
  finalize_workflow(best_tuned_early)

final_fit_early <- final_wf_early %>%
  last_fit(split_early, metrics = metrics)

# Evaluer
cat("\n--- Model 2: Test metrics ---\n")
collect_metrics(final_fit_early) %>% print()

collect_predictions(final_fit_early) %>%
  conf_mat(truth = early_churn, estimate = .pred_class)

collect_predictions(final_fit_early) %>%
  roc_curve(truth = early_churn, .pred_1) %>%
  autoplot() +
  labs(title = "ROC Curve — Early Churn Model")

plot_vip(final_fit_early, "What Drives Early Churn?")

##################################################
# SCENARIER
##################################################
churn_model <- extract_workflow(final_fit)
early_model <- extract_workflow(final_fit_early)

scenarier <- tibble(
  case = c("Best case", "Average case", "Worst case"),
  
  # Numeriske
  account_active_days      = c(60, 45, 9),
  age                      = c(60, 50, 30),
  previous_subscriptions   = c(5, 2, 0),
  previous_campaigns       = c(3, 1, 0),
  previous_trials          = c(1, 1, 0),
  newsletters_before_order = c(3, 1, 0),
  newsletters_after_order  = c(3, 1, 0),
  antal_sidevisninger      = c(300, 100, 0),
  antal_dage_aktiv         = c(25, 10, 0),
  andel_restricted         = c(0.35, 0.30, 0.0),
  avg_scroll_depth         = c(0.45, 0.35, 0.0),
  andel_mobil              = c(0.5, 0.5, 1.0),
  andel_social             = c(0.05, 0.05, 0.0),
  andel_email              = c(0.02, 0.01, 0.0),
  andel_search             = c(0.10, 0.10, 0.0),
  
  # Kategoriske
  koen = factor(
    c("Mand", "Mand", "Kvinde"),
    levels = levels(model_data_clean$koen)
  ),
  kundetid_gruppe = factor(
    c("6+ måneder", "6+ måneder", "0-7 dage"),
    levels = levels(model_data_clean$kundetid_gruppe)
  ),
  utm_content = factor(
    c("A", "A", "Andet"),
    levels = levels(model_data_clean$utm_content)
  ),
  utm_medium = factor(
    c("content", "content", "mail"),
    levels = levels(model_data_clean$utm_medium)
  ),
  utm_source = factor(
    c("jp.dk", "jp.dk", "mail"),
    levels = levels(model_data_clean$utm_source)
  ),
  permission_given_order = factor(
    c("1", "1", "0"),
    levels = levels(model_data_clean$permission_given_order)
  ),
  cluster_label = factor(
    c("Desktop-traditionalisten", "Desktop-traditionalisten", "Spøgelses-brugeren (Lav aktivitet)"),
    levels = levels(model_data_clean$cluster_label)
  ),
  
  # Dummy målvariable
  churn       = factor(c("0", "0", "0"), levels = c("1", "0")),
  early_churn = factor(c("0", "0", "0"), levels = c("0", "1"))
)

# Predict
p_churn <- predict(churn_model, scenarier, type = "prob") %>%
  select(churn_prob = .pred_1)

p_early <- predict(early_model, scenarier, type = "prob") %>%
  select(early_prob = .pred_1)

scenario_results <- bind_cols(scenarier, p_churn, p_early) %>%
  mutate(
    total_risk = churn_prob + (1 - churn_prob) * early_prob,
    across(c(churn_prob, early_prob, total_risk), ~ round(.x * 100, 1))
  ) %>%
  arrange(total_risk) %>%
  select(case, total_risk, churn_prob, early_prob)

cat("\n--- Scenarie resultater ---\n")
print(scenario_results, width = Inf)