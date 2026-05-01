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
    subscription_cancel_date = as.Date(subscription_cancel_date),
    first_campaign_day = dmy(first_campaign_day),
    last_campaign_day = dmy(last_campaign_day),
    expiration_date = as.Date(expiration_date)
  ) %>%
  mutate(
    churn = case_when(
      is.na(expiration_date) ~ 0,
      expiration_date <= last_campaign_day ~ 1,
      expiration_date > last_campaign_day ~ 0
    )
  ) %>%
  mutate(
    continued_subscription = 1 - churn
  ) %>%
  mutate(
    early_churn = case_when(
      continued_subscription == 1 &
        !is.na(expiration_date) &
        expiration_date <= last_campaign_day + 90 ~ 1,
      continued_subscription == 1 ~ 0,
      continued_subscription == 0 ~ 0
    )
  )

# 4. Feature engineering
model_data <- merged_data %>%
  mutate(
    # Parse datoer
    order_date  = as.Date(ymd_hms(order_date)),
    birthdate   = as.Date(parse_date_time(birthdate, orders = c("dmy", "ymd", "mdy"))),
    usr_created = dmy(usr_created),
    
    # Afledte variable
    age = as.numeric(import_date - birthdate) / 365,
    kundetid_dage = as.numeric(order_date - usr_created)
  ) %>%
  
  # Fjern ugyldige observationer
  filter(
    !is.na(birthdate),
    !is.na(usr_created),
    !is.na(order_date),
    previous_trials <= 100,
    age >= 15,
    age <= 105,
    kundetid_dage >= 0
  ) %>%
  
  # Rens kategoriske variable
  mutate(
    churn = factor(churn),
    type = fct_na_value_to_level(factor(type), "Ingen afmelding"),
    reason = fct_na_value_to_level(factor(reason), "Ingen afmelding"),
    koen = fct_recode(factor(koen), "Ikke oplyst" = ""),
    permission_given_order = factor(permission_given_order),
    permission_given_today = factor(permission_given_today),
    
    permission_given_order = factor(if_else(permission_given_order == "true", 1, 0)),
    permission_given_today = factor(if_else(permission_given_today == "true", 1, 0)),
    
    expiration_date = replace_na(expiration_date, as.Date("3000-01-01"))
  ) %>%
  
  # Kundelevetid grupper
  mutate(
    kundetid_gruppe = case_when(
      kundetid_dage <= 7 ~ "0-7 dage",
      kundetid_dage <= 30 ~ "8-30 dage",
      kundetid_dage <= 180 ~ "1-6 måneder",
      TRUE ~ "6+ måneder"
    ) %>% factor()
  ) %>%
  
  # Fjern hjælpe-variable og rå datoer
  select(
    -kundetid_dage,
    -order_date,
    -birthdate,
    -usr_created,
  )

# 5. Tjek resultatet
glimpse(model_data)

# Overblik over churn
model_data %>%
  count(churn) %>%
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

# Outlier tjek - dem har vi fjernet fra model_data
outliers <- merged_data %>%
  mutate(
    order_date  = as.Date(ymd_hms(order_date)),
    birthdate   = as.Date(parse_date_time(birthdate, orders = c("dmy", "ymd", "mdy"))),
    usr_created = dmy(usr_created),
    
    age = as.numeric(Sys.Date() - birthdate) / 365,
    kundetid_dage = as.numeric(order_date - usr_created)
  ) %>%
  filter(
    age < 15 | age > 105 |
      kundetid_dage < 0 |
      is.na(age) |
      is.na(kundetid_dage)
  ) %>%
  select(pseudo_id, birthdate, age, usr_created, order_date, kundetid_dage)

# Kig outliers
glimpse(outliers)

# Gem data
saveRDS(model_data, "data/model_data.rds")