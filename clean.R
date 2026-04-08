pacman::p_load(tidyverse, lubridate, forcats)

# Indlæs data
cancellation <- read.csv("data/cancellation.csv")
subscription <- read.csv("data/subscription_v2.csv", sep = ";")
behavior <- read.csv("data/behavior.csv")

# Fjern dubletter
cancellation <- cancellation %>% distinct(pseudo_id, .keep_all = TRUE)
subscription <- subscription %>% distinct(pseudo_id, .keep_all = TRUE)

# Flet data + churn-definition
merged_data <- subscription %>%
  left_join(cancellation, by = "pseudo_id") %>%
  
  # Parse dates
  mutate(
    subscription_cancel_date = dmy(subscription_cancel_date),
    first_campaign_day = dmy(first_campaign_day),
    last_campaign_day = dmy(last_campaign_day),
    expiration_date = ymd(expiration_date)
  ) %>%
  
  # Definer churn efter kampagneperioden
  mutate(
    churn = if_else(
      !is.na(subscription_cancel_date) &
        subscription_cancel_date < as.Date("2999-01-01") &
        subscription_cancel_date > last_campaign_day,
      1, 0
    )
  )

# Feature engineering / rensning
model_data <- merged_data %>%
  select(-pseudo_id) %>%
  
  mutate(
    order_date = as.Date(ymd_hms(order_date)),
    birthdate = as.Date(parse_date_time(birthdate, orders = c("dmy", "ymd", "mdy"))),
    usr_created = dmy(usr_created)
  ) %>%
  
  # Fjern NA
  filter(!is.na(birthdate), !is.na(usr_created), !is.na(order_date)) %>%
  
  # Kategorisk rensning
  mutate(
    churn = factor(churn),
    type = factor(type),
    reason = factor(reason),
    koen = factor(koen),
    type = fct_na_value_to_level(type, level = "Ingen afmelding"),
    reason = fct_na_value_to_level(reason, level = "Ingen afmelding"),
    expiration_date = replace_na(expiration_date, as.Date("3000-01-01"))
  ) %>%
  
  # Samtykker
  mutate(
    permission_given_order = factor(if_else(permission_given_order == "true", 1, 0)),
    permission_given_today = factor(if_else(permission_given_today == "true", 1, 0))
  ) %>%
  
  # Afledte variable
  mutate(
    age = as.numeric(Sys.Date() - birthdate) / 365,
    kundetid_dage = as.numeric(order_date - usr_created)
  ) %>%
  
  # Ekstra sikkerhed
  filter(!is.na(kundetid_dage)) %>%
  
  # Kundetid grupper
  mutate(
    kundetid_gruppe = case_when(
      kundetid_dage <= 7 ~ "0-7 dage",
      kundetid_dage <= 30 ~ "8-30 dage",
      kundetid_dage <= 180 ~ "1-6 måneder",
      TRUE ~ "6+ måneder"
    ) %>% factor()
  )

# Fjern ubrugte kolonner
model_data <- model_data %>%
  select(
    -kundetid_dage,
    -order_date,
    -birthdate,
    -usr_created
  )

# Tjek
glimpse(model_data)
summary(model_data$age)
colSums(is.na(model_data))

# Churn distribution
model_data %>%
  count(churn) %>%
  mutate(prop = n / sum(n))

# Gem renset data
saveRDS(model_data, "data/model_data.rds")