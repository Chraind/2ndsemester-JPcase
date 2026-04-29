# ----------------------------
# 1. Load packages
# ----------------------------
pacman::p_load(tidyverse, DataExplorer, ggpubr)

# ----------------------------
# 2. Load data
# ----------------------------
behavior <- read.csv("data/behavior.csv")
model_data <- readRDS("data/model_data.rds")

glimpse(behavior)

# ----------------------------
# 3. Create behavioral customer-level features
# ----------------------------
customer_behavior <- behavior %>%
  group_by(pseudo_id) %>%
  summarise(
    n_visits = n(),
    avg_scroll = mean(scroll_depth, na.rm = TRUE),
    share_mobile = mean(dvce_type == "Mobile"),
    share_desktop = mean(dvce_type == "Computer"),
    share_restricted = mean(page_restricted == "yes"),
    share_search = mean(refr_medium == "search"),
    share_internal = mean(refr_medium == "internal"),
    n_unique_pages = n_distinct(webpage_id)
  )

saveRDS(customer_behavior, "data/coolbehavior.rds")

# ----------------------------
# 4. Prepare subscription features
# ----------------------------
subscription_features <- model_data %>%
  select(
    pseudo_id,
    account_active_days,
    previous_subscriptions,
    previous_campaigns,
    previous_trials,
    newsletters_before_order,
    newsletters_after_order,
    churn
  )

glimpse(subscription_features)
# ----------------------------
# 5. MERGE DATA
# ----------------------------
customer_df <- subscription_features %>%
  left_join(customer_behavior, by = "pseudo_id")

# Indsæt 0 i stedet for NA for de 70 "Spøgelses-brugere" 
customer_df <- customer_df %>%
  mutate(across(
    c(n_visits, n_unique_pages, avg_scroll, starts_with("share_")),
    ~replace_na(.x, 0)
  ))

# sanity check - should now be 1,275
nrow(customer_df)

# check
colSums(is.na(customer_df))

# ----------------------------
# 6. Remove ID + handle missing values
# ----------------------------
cluster_input <- customer_df %>%
  select(-pseudo_id, -churn) 

# check
glimpse(cluster_input)

# ----------------------------
# 7. Explore distributions
# ----------------------------
plot_histogram(cluster_input)

# ----------------------------
# 8. Scale variables
# ----------------------------
customer_scaled <- scale(cluster_input)

# ----------------------------
# 9. PCA
# ----------------------------
pca <- prcomp(customer_scaled, scale = TRUE)

summary(pca)

screeplot(pca, type = "line")
abline(h = 1, col = "red", lty = 3)

biplot(pca, scale = 0)

# profilér på oprindelige Y variabler

# ----------------------------
# 10. Hierarchical clustering
# ----------------------------
# hc_complete <- hclust(dist(pca$x[,1:4]), method = "complete")
# plot(hc_complete)
# 
# customer_df <- customer_df %>%
#   filter(complete.cases(select(., -pseudo_id, -churn)))
# 
# customer_df$cluster_hc <- cutree(hc_complete, 5)
# 
# table(customer_df$cluster_hc)

hc_ward <- hclust(dist(pca$x[,1:4]), method = "ward.D2")
plot(hc_ward)

customer_df <- customer_df %>%
  filter(complete.cases(select(., -pseudo_id, -churn)))

customer_df$cluster_hc <- cutree(hc_ward, 5)

table(customer_df$cluster_hc)

# ----------------------------
# 11. K-means clustering
# ----------------------------
set.seed(8)

km <- kmeans(pca$x[,1:4], centers = 5, nstart = 20)

customer_df$cluster_km <- km$cluster

table(customer_df$cluster_km)

# compare methods
table(customer_df$cluster_km, customer_df$cluster_hc)

# ----------------------------
# 12. Cluster profiling
# ----------------------------
customer_df$cluster_km <- as.factor(customer_df$cluster_km)

cluster_profile <- customer_df %>%
  group_by(cluster_km) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))

# ----------------------------
# 13. Visualisation
# ----------------------------
ggboxplot(customer_df, x = "cluster_km", y = "n_visits",
          color = "cluster_km")

ggboxplot(customer_df, x = "cluster_km", y = "avg_scroll",
          color = "cluster_km")

ggboxplot(customer_df, x = "cluster_km", y = "account_active_days",
          color = "cluster_km")

ggboxplot(customer_df, x = "cluster_km", y = "previous_subscriptions",
          color = "cluster_km")

# ----------------------------
# 14. ANOVA tests
# ----------------------------
summary(aov(n_visits ~ cluster_km, data = customer_df))
summary(aov(avg_scroll ~ cluster_km, data = customer_df))

summary(aov(account_active_days ~ cluster_km, data = customer_df))
summary(aov(previous_subscriptions ~ cluster_km, data = customer_df))
summary(aov(previous_campaigns ~ cluster_km, data = customer_df))
summary(aov(previous_trials ~ cluster_km, data = customer_df))
summary(aov(newsletters_before_order ~ cluster_km, data = customer_df))
summary(aov(newsletters_after_order ~ cluster_km, data = customer_df))

# ----------------------------
# 15. CHURN ANALYSIS (IMPORTANT)
# ----------------------------
churn_by_cluster <- customer_df %>%
  group_by(cluster_km) %>%
  summarise(
    churn_rate = mean(churn == 1, na.rm = TRUE),
    n = n()
  )

print(churn_by_cluster)

# ----------------------------
# PRINT CLUSTER MEANS (COORDINATES)
# ----------------------------
cluster_summary_long <- customer_df %>%
  group_by(cluster_km) %>%
  summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE))) %>%
  pivot_longer(cols = -cluster_km, names_to = "variable", values_to = "mean_value") %>%
  pivot_wider(names_from = cluster_km, names_prefix = "Cluster_", values_from = mean_value)

# This will print the full table in your console
print(cluster_summary_long, n = 50)

# Definér 5 danske labels baseret på de opdaterede profiler
customer_df <- customer_df %>%
  mutate(cluster_label = case_when(
    cluster_km == 1 ~ "Power-brugeren (Mest mobil)",
    cluster_km == 2 ~ "Desktop-traditionalisten",
    cluster_km == 3 ~ "Veteranen (Høj historik)",
    cluster_km == 4 ~ "Mobil-nykommeren",
    cluster_km == 5 ~ "Spøgelses-brugeren (Lav aktivitet)",
    TRUE ~ "Andet"
  ))

# Tjek den nye fordeling
table(customer_df$cluster_label, customer_df$churn)

# Gem mapping til din Tidymodels-fil
cluster_mapping <- customer_df %>% 
  select(pseudo_id, cluster_label)

saveRDS(cluster_mapping, "data/cluster_mapping.rds")

# Tjek fordelingen
table(customer_df$cluster_label)

