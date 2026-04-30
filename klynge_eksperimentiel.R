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
# 3. Create behavioral customer-level features (Enhanced)
# ----------------------------
# Set a reference date for recency calculations
max_date <- as.Date(max(behavior$collector_tstamp))

customer_behavior <- behavior %>%
  mutate(
    date = as.Date(collector_tstamp),
    is_recent = date > (max_date - 7)
  ) %>%
  group_by(pseudo_id) %>%
  summarise(
    # TOTALS
    n_visits = n(),
    n_unique_pages = n_distinct(webpage_id),
    
    # RECENCY (Days since last seen)
    days_since_last_visit = as.numeric(max_date - max(date)),
    
    # VELOCITY (Activity trend)
    recent_visit_count = sum(is_recent),
    visit_velocity = recent_visit_count / (n_visits / as.numeric(max_date - min(date) + 1) * 7 + 1),
    
    # ENGAGEMENT QUALITY
    avg_scroll = mean(scroll_depth, na.rm = TRUE),
    recent_scroll = mean(scroll_depth[is_recent], na.rm = TRUE),
    scroll_dropoff = recent_scroll / (avg_scroll + 0.01),
    
    # CONTEXT & INTENSITY
    share_mobile = mean(dvce_type == "Mobile"),
    share_desktop = mean(dvce_type == "Computer"),
    share_restricted = mean(page_restricted == "yes"),
    pages_per_visit = n_unique_pages / (n_visits + 1)
  ) %>%
  # Handle internal NAs from users with 0 recent visits
  mutate(across(c(recent_scroll, scroll_dropoff), ~replace_na(., 0)))

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

# ----------------------------
# 5. MERGE & FIX GHOST USERS (CRITICAL FIX)
# ----------------------------
customer_df <- subscription_features %>%
  left_join(customer_behavior, by = "pseudo_id")

# Indsæt værdier i stedet for NA for de brugere der ikke findes i behavior.csv
customer_df <- customer_df %>%
  mutate(
    # Counts and shares get 0
    across(c(n_visits, n_unique_pages, avg_scroll, recent_visit_count, 
             recent_scroll, scroll_dropoff, visit_velocity,
             share_mobile, share_desktop, share_restricted, pages_per_visit), 
           ~replace_na(.x, 0)),
    
    # Recency gets a 'Penalty' value (Max days + 1)
    days_since_last_visit = replace_na(days_since_last_visit, max(days_since_last_visit, na.rm = TRUE) + 1)
  ) %>%
  # Final safety drop to ensure PCA doesn't fail
  drop_na()

# Sanity checks
print(paste("Rows after merge:", nrow(customer_df)))
print("Missing values per column:")
print(colSums(is.na(customer_df)))

# ----------------------------
# 6. Remove ID + handle missing values
# ----------------------------
cluster_input <- customer_df %>%
  select(-pseudo_id, -churn) 

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

# ----------------------------
# 10. Hierarchical clustering
# ----------------------------
# Using first 5 components to capture the new velocity/recency variance
hc_ward <- hclust(dist(pca$x[,1:5]), method = "ward.D2")
plot(hc_ward)

customer_df$cluster_hc <- cutree(hc_ward, 5)

# ----------------------------
# 11. K-means clustering
# ----------------------------
set.seed(8)
km <- kmeans(pca$x[,1:5], centers = 5, nstart = 20)
customer_df$cluster_km <- km$cluster

# ----------------------------
# 12. Cluster profiling
# ----------------------------
customer_df$cluster_km <- as.factor(customer_df$cluster_km)

# ----------------------------
# 13-14. Visuals & ANOVA
# ----------------------------
# Check n_visits across clusters
ggboxplot(customer_df, x = "cluster_km", y = "n_visits", color = "cluster_km")
# Check the new Recency feature
ggboxplot(customer_df, x = "cluster_km", y = "days_since_last_visit", color = "cluster_km")

summary(aov(days_since_last_visit ~ cluster_km, data = customer_df))

# ----------------------------
# 15. CHURN ANALYSIS & LABELS
# ----------------------------
churn_by_cluster <- customer_df %>%
  group_by(cluster_km) %>%
  summarise(
    churn_rate = mean(churn == 1, na.rm = TRUE),
    avg_recency = mean(days_since_last_visit),
    n = n()
  )

print(churn_by_cluster)

# Finalize Labels (Adjust names based on cluster_profile output)
customer_df <- customer_df %>%
  mutate(cluster_label = case_when(
    cluster_km == 1 ~ "Aktive Power-brugere",
    cluster_km == 2 ~ "Desktop-traditionalister",
    cluster_km == 3 ~ "Loyale Veteraner",
    cluster_km == 4 ~ "Nye/Ustabile Mobil-brugere",
    cluster_km == 5 ~ "Inaktive Spøgelses-brugere",
    TRUE ~ "Andet"
  ))

# Save mapping
cluster_mapping <- customer_df %>% 
  select(pseudo_id, cluster_label)

saveRDS(cluster_mapping, "data/cluster_mapping.rds")
table(customer_df$cluster_label)