# Indlæs nødvendige pakker
pacman::p_load(tidyverse, DataExplorer, ggpubr)

# Indlæs data
behavior <- read.csv("data/behavior.csv")
model_data <- readRDS("data/model_data.rds")

glimpse(model_data)

table_data <- model_data %>%
  select(
    account_active_days,
    previous_subscriptions,
    previous_campaigns,
    previous_trials,
    newsletters_before_order,
    newsletters_after_order
  )

view(table_data)

# Omdan data til kundeniveau (én række pr. bruger)
customer_df <- behavior %>%
  group_by(pseudo_id) %>%
  summarise(
    n_visits = n(),  # antal besøg
    avg_scroll = mean(scroll_depth, na.rm = TRUE),  # gennemsnitlig scroll depth
    share_mobile = mean(dvce_type == "Mobile"),  # andel mobilbrug
    share_desktop = mean(dvce_type == "Computer"),  # andel desktop
    share_restricted = mean(page_restricted == "yes"),  # andel paywall-indhold
    share_search = mean(refr_medium == "search"),  # andel trafik fra search
    share_internal = mean(refr_medium == "internal"),  # andel intern navigation
    n_unique_pages = n_distinct(webpage_id) # antal unikke sider
  )

# Undersøg fordelingen af variabler
plot_histogram(customer_df[,-1])

# Standardiser variabler før PCA
customer_scaled <- scale(customer_df[,-1])

# Udfør PCA
pca <- prcomp(customer_scaled, scale = TRUE)

# Se hvor meget variation komponenterne forklarer
summary(pca)

# Screeplot til valg af antal komponenter
screeplot(pca, type = "line")
abline(h = 1, col = "red", lty = 3)

# Visualisering af PCA
biplot(pca, scale = 0)

# Hierarkisk klyngeanalyse på de første 4 komponenter
hc_complete <- hclust(dist(pca$x[,1:4]), method = "complete")
plot(hc_complete)

# Opdel i 3 klynger
clusters_hc <- cutree(hc_complete, 3)
customer_df$cluster_hc <- clusters_hc

# Se fordeling af observationer
table(customer_df$cluster_hc)

# K-means klyngeanalyse (k = 3)
set.seed(1)
km <- kmeans(pca$x[,1:4], centers = 3, nstart = 20)

customer_df$cluster_km <- km$cluster

# Fordeling af klynger
table(customer_df$cluster_km)

# Sammenlign med hierarkisk clustering
table(customer_df$cluster_km, customer_df$cluster_hc)

# Konverter til faktor (vigtigt for ANOVA)
customer_df$cluster_km <- as.factor(customer_df$cluster_km)

# Beregn gennemsnit pr. klynge (profilering)
cluster_profile <- customer_df %>%
  group_by(cluster_km) %>%
  summarise(across(-pseudo_id, mean))

print(cluster_profile)

# Visualisering af forskelle mellem klynger
ggboxplot(customer_df, x = "cluster_km", y = "n_visits",
          color = "cluster_km")

ggboxplot(customer_df, x = "cluster_km", y = "avg_scroll",
          color = "cluster_km")

ggboxplot(customer_df, x = "cluster_km", y = "share_mobile",
          color = "cluster_km")

ggboxplot(customer_df, x = "cluster_km", y = "share_search",
          color = "cluster_km")

# ANOVA-test for forskelle mellem klynger
summary(aov(n_visits ~ cluster_km, data = customer_df))
summary(aov(avg_scroll ~ cluster_km, data = customer_df))
summary(aov(share_mobile ~ cluster_km, data = customer_df))
summary(aov(share_search ~ cluster_km, data = customer_df))
summary(aov(n_unique_pages ~ cluster_km, data = customer_df))
