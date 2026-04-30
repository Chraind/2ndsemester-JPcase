pacman::p_load(tidyverse, lubridate, forcats, readr)

# 1. Indlæs data
cancellation <- read.csv("data/cancellation.csv")
subscription <- read.csv("data/subscription_v2.csv", sep = ";")

# Sæt import_date baseret på nyeste data i filen
import_date <- as.Date("2026-03-24")

# 2. Behold seneste observation pr. bruger
cancellation <- cancellation %>%
  mutate(expiration_date = ymd(expiration_date)) %>%
  group_by(pseudo_id) %>%
  slice_max(expiration_date, n = 1, with_ties = FALSE) %>%
  ungroup()

subscription <- subscription %>%
  mutate(subscription_cancel_date = dmy(subscription_cancel_date)) %>%
  group_by(pseudo_id) %>%
  slice_max(subscription_cancel_date, n = 1, with_ties = FALSE) %>%
  ungroup()

# 3. Merge + churn logik
merged_data <- subscription %>%
  left_join(cancellation, by = "pseudo_id") %>%
  mutate(
    # Date formatting
    order_date = as.Date(ymd_hms(order_date)),
    expiration_date = as.Date(expiration_date),
    subscription_cancel_date = as.Date(subscription_cancel_date),
    
    # Define the 60-day campaign window
    campaign_end = order_date + 60
  ) %>%
  mutate(
    # Churn: 1 if they expire within the 60 days
    churn = case_when(
      is.na(expiration_date) ~ 0,
      expiration_date <= campaign_end ~ 1,
      expiration_date > campaign_end ~ 0
    )
  ) %>%
  mutate(
    # Logical opposite of churn
    continued_subscription = 1 - churn
  ) %>%
  mutate(
    # Early Churn: Users who survived the 60 days but expired within 90 days after the campaign ended
    early_churn = case_when(
      continued_subscription == 1 & 
        !is.na(expiration_date) & 
        expiration_date <= (campaign_end + 90) ~ 1,
      TRUE ~ 0
    )
  )

# 4. Feature engineering - LEAKAGE-FREE HISTORY & OUTLIER REMOVAL
model_data <- merged_data %>%
  mutate(
    # A. Standardize dates (order_date is already a Date from section 3)
    usr_created  = dmy(usr_created),
    birthdate    = as.Date(parse_date_time(birthdate, orders = c("dmy", "ymd", "mdy"))),
    
    # B. The "Time Machine" Adjustment
    # Using campaign_end (order_date + 60) as the reference point
    days_to_remove = as.numeric(import_date - campaign_end),
    
    adj_active_days = case_when(
      is.na(expiration_date) ~ account_active_days - days_to_remove,
      expiration_date > campaign_end ~ account_active_days - as.numeric(expiration_date - campaign_end),
      TRUE ~ as.numeric(account_active_days)
    ),
    
    # Final active days (capped at 0)
    account_active_days = pmax(0, adj_active_days),
    
    # C. Seniority (How long were they customers BEFORE this specific campaign?)
    seniority_at_start = as.numeric(as.Date(order_date) - usr_created),
    
    # D. Age
    age = as.numeric(import_date - birthdate) / 365
  ) %>%
  # E. FILTERING: Remove negative seniority, extreme ages, and trial outliers
  filter(
    seniority_at_start >= 0, 
    age >= 15, 
    age <= 105,
    previous_trials <= 100
  ) %>%
  mutate(
    churn = factor(churn),
    type = fct_na_value_to_level(factor(type), "Ingen afmelding"),
    reason = fct_na_value_to_level(factor(reason), "Ingen afmelding"),
    koen = fct_recode(factor(koen), "Ikke oplyst" = ""),
    permission_given_order = factor(permission_given_order),
    permission_given_today = factor(permission_given_today),
    
    # Create a cleaner seniority group for the model
    customer_segment = case_when(
      seniority_at_start == 0 ~ "New Lead",
      seniority_at_start <= 365 ~ "Existing < 1yr",
      TRUE ~ "Loyal 1yr+"
    ) %>% factor()
  ) %>%
  # Clean up temporary adjustment columns
  select(
    -birthdate, 
    -usr_created,
    -adj_active_days,
    -days_to_remove
  )

# 5. Tjek resultatet
glimpse(model_data)

# Overblik over churn
model_data %>%
  count(churn) %>%
  mutate(prop = n / sum(n))

# Overblik over churn
model_data %>%
  count(early_churn) %>%
  mutate(prop = n / sum(n))

summary(model_data$account_active_days)
summary(subscription$account_active_days)

model_data %>%
  count(account_active_days, sort = TRUE) %>%
  head(20)

model_data %>%
  group_by(churn) %>%
  summarise(
    mean = mean(account_active_days),
    median = median(account_active_days),
    sd = sd(account_active_days)
  )

# Check for missing values
colSums(is.na(model_data))

# --- 5. Outlier Check (Cleaning up the parsing warning) ---
outliers <- merged_data %>%
  mutate(
    # Use the dates already formatted in merged_data
    # No need to call ymd_hms() again here
    birthdate = as.Date(parse_date_time(birthdate, orders = c("dmy", "ymd", "mdy"))),
    usr_created = dmy(usr_created),
    
    # Calculate age and seniority for checking
    age = as.numeric(import_date - birthdate) / 365,
    kundetid_dage = as.numeric(order_date - usr_created)
  ) %>%
  filter(
    age < 15 | age > 105 |
      kundetid_dage < 0 |
      is.na(age) |
      is.na(kundetid_dage)
  ) %>%
  select(pseudo_id, birthdate, age, usr_created, order_date, kundetid_dage)

# --- 6. Final Status Check ---
message(paste("Rows in model_data:", nrow(model_data)))
message(paste("Outliers removed:", nrow(outliers)))

# Gem data
saveRDS(model_data, "data/model_data.rds")

# Gem data
saveRDS(model_data, "data/model_data.rds")
