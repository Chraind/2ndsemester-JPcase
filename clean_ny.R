pacman::p_load(tidyverse, lubridate, forcats, readr)

# indlæs data
cancellation <- read_csv("data/cancellation.csv")
subscription <- read_delim("data/subscription_v2.csv", delim = ";")

# behold seneste observation per bruger
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

# flet data og lav variable
model_data <- subscription %>%
  left_join(cancellation, by = "pseudo_id") %>%
  mutate(
    # datoer
    first_campaign_day = dmy(first_campaign_day),
    last_campaign_day = dmy(last_campaign_day),
    order_date = as.Date(ymd_hms(order_date)),
    birthdate = as.Date(parse_date_time(birthdate, c("dmy", "ymd", "mdy"))),
    usr_created = dmy(usr_created),
    
    # churn-variable
    churn = if_else(!is.na(expiration_date) & expiration_date <= last_campaign_day, 1, 0),
    continued_subscription = 1 - churn,
    early_churn = if_else(
      continued_subscription == 1 &
        !is.na(expiration_date) &
        expiration_date <= last_campaign_day + 90,
      1, 0
    ),
    
    # afledte variable
    age = as.numeric(Sys.Date() - birthdate) / 365,
    kundetid_dage = as.numeric(order_date - usr_created),
    
    # rens kategoriske variable
    churn = factor(churn),
    koen = fct_recode(factor(koen), "ikke oplyst" = ""),
    type = fct_na_value_to_level(factor(type), "ingen afmelding"),
    reason = fct_na_value_to_level(factor(reason), "ingen afmelding"),
    
    # samtykkevariable
    permission_given_order = factor(if_else(permission_given_order == "true", 1, 0)),
    permission_given_today = factor(if_else(permission_given_today == "true", 1, 0)),
    
    # erstat manglende udløbsdato
    expiration_date = replace_na(expiration_date, as.Date("3000-01-01")),
    
    # grupper kundetid
    kundetid_gruppe = case_when(
      kundetid_dage <= 7 ~ "0-7 dage",
      kundetid_dage <= 30 ~ "8-30 dage",
      kundetid_dage <= 180 ~ "1-6 måneder",
      TRUE ~ "6+ måneder"
    ) %>% factor()
  ) %>%
  filter(
    !is.na(birthdate),
    !is.na(usr_created),
    !is.na(order_date),
    between(age, 15, 105),
    kundetid_dage >= 0
  ) %>%
  select(-kundetid_dage, -order_date, -birthdate, -usr_created)

# tjek data
glimpse(model_data)
summary(model_data$age)
colSums(is.na(model_data))

model_data %>%
  count(churn) %>%
  mutate(prop = n / sum(n))

# gem data
saveRDS(model_data, "data/model_data.rds")
