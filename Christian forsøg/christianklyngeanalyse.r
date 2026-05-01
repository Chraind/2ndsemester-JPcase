pacman::p_load(tidyverse, DataExplorer, ggpubr)

# ── 1. Indlæs data ─────────────────────────────────────────────────────────────
behavior   <- read.csv("data/behavior.csv")
model_data <- readRDS("data/model_data.rds")

# ── 2. Behavioral features ─────────────────────────────────────────────────────
customer_behavior <- behavior %>%
  group_by(pseudo_id) %>%
  summarise(
    n_visits         = n(),
    avg_scroll       = mean(scroll_depth, na.rm = TRUE),
    share_mobile     = mean(dvce_type == "Mobile"),
    share_desktop    = mean(dvce_type == "Computer"),
    share_restricted = mean(page_restricted == "yes"),
    share_search     = mean(refr_medium == "search"),
    share_internal   = mean(refr_medium == "internal"),
    n_unique_pages   = n_distinct(webpage_id)
  )

saveRDS(customer_behavior, "data/coolbehavior.rds")

# ── 3. Subscription features ───────────────────────────────────────────────────
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

# ── 4. Merge ───────────────────────────────────────────────────────────────────
customer_df <- subscription_features %>%
  left_join(customer_behavior, by = "pseudo_id") %>%
  mutate(across(
    c(n_visits, n_unique_pages, avg_scroll, starts_with("share_")),
    ~ replace_na(.x, 0)
  ))

cat("Rækker:", nrow(customer_df), "\n")
colSums(is.na(customer_df))

# ── 5. Cluster input ───────────────────────────────────────────────────────────
cluster_input <- customer_df %>%
  select(-pseudo_id, -churn)

# ── 6. Udforsk fordelinger ─────────────────────────────────────────────────────
plot_histogram(cluster_input)

# ── 7. Skalér ──────────────────────────────────────────────────────────────────
customer_scaled <- scale(cluster_input)

# ── 8. PCA ─────────────────────────────────────────────────────────────────────
pca <- prcomp(customer_scaled, scale = TRUE)
summary(pca)

screeplot(pca, type = "line")
abline(h = 1, col = "red", lty = 3)
biplot(pca, scale = 0)

# ── 9. Hierarchical clustering ─────────────────────────────────────────────────
hc_ward <- hclust(dist(pca$x[, 1:4]), method = "ward.D2")
plot(hc_ward)

customer_df <- customer_df %>%
  filter(complete.cases(select(., -pseudo_id, -churn)))

customer_df$cluster_hc <- cutree(hc_ward, 5)
table(customer_df$cluster_hc)

# ── 10. K-means ────────────────────────────────────────────────────────────────
set.seed(8)
km <- kmeans(pca$x[, 1:4], centers = 5, nstart = 20)
customer_df$cluster_km <- km$cluster

table(customer_df$cluster_km)
table(customer_df$cluster_km, customer_df$cluster_hc)

# ── 11. Profiling ──────────────────────────────────────────────────────────────
customer_df$cluster_km <- as.factor(customer_df$cluster_km)

cluster_profile <- customer_df %>%
  group_by(cluster_km) %>%
  summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE)))

# ── 12. Visualisering ──────────────────────────────────────────────────────────
ggboxplot(customer_df, x = "cluster_km", y = "n_visits",         color = "cluster_km")
ggboxplot(customer_df, x = "cluster_km", y = "avg_scroll",       color = "cluster_km")
ggboxplot(customer_df, x = "cluster_km", y = "account_active_days", color = "cluster_km")
ggboxplot(customer_df, x = "cluster_km", y = "previous_subscriptions", color = "cluster_km")

# ── 13. ANOVA ──────────────────────────────────────────────────────────────────
summary(aov(n_visits                  ~ cluster_km, data = customer_df))
summary(aov(avg_scroll                ~ cluster_km, data = customer_df))
summary(aov(account_active_days       ~ cluster_km, data = customer_df))
summary(aov(previous_subscriptions    ~ cluster_km, data = customer_df))
summary(aov(previous_campaigns        ~ cluster_km, data = customer_df))
summary(aov(previous_trials           ~ cluster_km, data = customer_df))
summary(aov(newsletters_before_order  ~ cluster_km, data = customer_df))
summary(aov(newsletters_after_order   ~ cluster_km, data = customer_df))

# ── 14. Churn per cluster ──────────────────────────────────────────────────────
churn_by_cluster <- customer_df %>%
  group_by(cluster_km) %>%
  summarise(
    churn_rate = mean(as.numeric(as.character(churn)) == 1, na.rm = TRUE),
    n = n()
  )
print(churn_by_cluster)

# ── 15. Cluster means ──────────────────────────────────────────────────────────
cluster_summary_long <- customer_df %>%
  group_by(cluster_km) %>%
  summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE))) %>%
  pivot_longer(cols = -cluster_km, names_to = "variable", values_to = "mean_value") %>%
  pivot_wider(names_from = cluster_km, names_prefix = "Cluster_", values_from = mean_value)

print(cluster_summary_long, n = 50)

# ── 16. Labels baseret på faktiske profiler ────────────────────────────────────
# Baseret på cluster_summary_long:
# Cluster 1: 623 besøg, 69% mobil          → Power-brugeren
# Cluster 2: 129 besøg, 91% desktop        → Desktop-traditionalisten
# Cluster 3: 75 besøg,  90% mobil          → Mobil-nykommeren
# Cluster 4: 218 besøg, 12.9 sub.          → Veteranen
# Cluster 5: 4.5 besøg, næsten ingen aktiv → Spøgelses-brugeren

customer_df <- customer_df %>%
  mutate(cluster_label = case_when(
    cluster_km == 1 ~ "Power-brugeren (Mest mobil)",
    cluster_km == 2 ~ "Desktop-traditionalisten",
    cluster_km == 3 ~ "Mobil-nykommeren",
    cluster_km == 4 ~ "Veteranen (Høj historik)",
    cluster_km == 5 ~ "Spøgelses-brugeren (Lav aktivitet)",
    TRUE ~ "Andet"
  ))

table(customer_df$cluster_label, customer_df$churn)
table(customer_df$cluster_label)

# ── 17. Gem cluster_mapping UDEN dubletter ─────────────────────────────────────
cluster_mapping <- customer_df %>%
  select(pseudo_id, cluster_label) %>%
  distinct(pseudo_id, .keep_all = TRUE)  # ← løser many-to-many problemet

saveRDS(cluster_mapping, "data/cluster_mapping.rds")

cat("cluster_mapping gemt:", nrow(cluster_mapping), "unikke kunder\n")