pacman::p_load(tidyverse, lubridate, forcats, readr)

# ── 1. Indlæs data ─────────────────────────────────────────────────────────────
import_date   <- as.Date("2026-03-10")
kampagne_slut <- as.Date("2025-05-23")

subcription  <- read.csv("data/subscription_v2.csv", sep = ";", encoding = "UTF-8")
cancellation <- read.csv("data/cancellation.csv",    encoding = "UTF-8")
behavior     <- read.csv("data/behavior.csv",        encoding = "UTF-8")

# ── 2. Konverter datoer i subscription ────────────────────────────────────────
subcription <- subcription %>%
  mutate(
    first_campaign_day = dmy(first_campaign_day),
    last_campaign_day  = dmy(last_campaign_day),
    order_date         = ymd_hms(order_date),
    birthdate          = dmy(birthdate),
    usr_created        = dmy(usr_created)
  )

# ── 3. Merge + churn logik ─────────────────────────────────────────────────────
cancellation <- cancellation %>%
  mutate(expiration_date = ymd(expiration_date)) %>%
  group_by(pseudo_id) %>%
  slice_max(expiration_date, n = 1, with_ties = FALSE) %>%
  ungroup()

merged_data <- subcription %>%
  left_join(cancellation, by = "pseudo_id") %>%
  mutate(
    churn = case_when(
      is.na(expiration_date)               ~ 0,
      expiration_date <= last_campaign_day ~ 1,
      TRUE                                 ~ 0
    ),
    continued_subscription = 1 - churn,
    early_churn = case_when(
      continued_subscription == 1 &
        !is.na(expiration_date) &
        expiration_date <= kampagne_slut + 90 ~ 1,
      TRUE                                    ~ 0
    )
  )

# ── 4. Feature engineering ─────────────────────────────────────────────────────
model_data <- merged_data %>%
  mutate(
    order_date  = as.Date(ymd_hms(order_date)),
    birthdate   = as.Date(parse_date_time(birthdate, orders = c("dmy", "ymd", "mdy"))),
    usr_created = as.Date(usr_created),
    aktivitet_slut = pmin(
      if_else(is.na(expiration_date), last_campaign_day, expiration_date),
      last_campaign_day
    ),
    account_active_days = pmax(0, as.numeric(aktivitet_slut - first_campaign_day)),
    age           = as.numeric(import_date - birthdate) / 365,
    kundetid_dage = as.numeric(order_date - usr_created)
  ) %>%
  filter(
    !is.na(birthdate), !is.na(usr_created), !is.na(order_date),
    previous_trials <= 50,
    age >= 15, age <= 105,
    kundetid_dage >= 0
  ) %>%
  mutate(
    churn = factor(churn),
    koen  = factor(case_when(
      is.na(koen) | koen == "" ~ "Ikke oplyst",
      TRUE                     ~ koen
    )),
    permission_given_order = factor(if_else(permission_given_order == "true", 1, 0)),
    kundetid_gruppe = case_when(
      kundetid_dage <= 7   ~ "0-7 dage",
      kundetid_dage <= 30  ~ "8-30 dage",
      kundetid_dage <= 180 ~ "1-6 måneder",
      TRUE                 ~ "6+ måneder"
    ) %>% factor(),
    utm_content = str_extract(order_trackertag, "(?<=utm_content=)[^&]+"),
    utm_medium  = str_extract(order_trackertag, "(?<=utm_medium=)[^&]+"),
    utm_source  = str_extract(order_trackertag, "(?<=utm_source=)[^&]+"),
    utm_content = toupper(utm_content),
    utm_content = case_when(
      utm_content %in% c("A", "B", "C", "D") ~ utm_content,
      is.na(utm_content)                      ~ "Ukendt",
      TRUE                                    ~ "Andet"
    ) %>% factor(),
    utm_medium = factor(utm_medium),
    utm_source = factor(utm_source)
  ) %>%
  select(
    -kundetid_dage, -order_date, -birthdate, -usr_created,
    -aktivitet_slut, -order_trackertag, -expiration_date,
    -subscription_cancel_date, -permission_given_today,
    -first_campaign_day, -last_campaign_day,
    -type, -reason
  )
# ── 5. Behavior features (udvidet) ────────────────────────────────────────────
behavior_features <- behavior %>%
  mutate(
    dt    = as.Date(dt),
    hour  = as.integer(substr(collector_tstamp, 12, 13))
  ) %>%
  group_by(pseudo_id) %>%
  summarise(
    # ── Eksisterende ──────────────────────────────────────────────────────────
    antal_sidevisninger = n(),
    antal_dage_aktiv    = n_distinct(dt),
    andel_restricted    = mean(page_restricted == "yes", na.rm = TRUE),
    avg_scroll_depth    = mean(scroll_depth, na.rm = TRUE),
    andel_mobil         = mean(dvce_type == "Mobile", na.rm = TRUE),
    andel_social        = mean(refr_medium == "social", na.rm = TRUE),
    andel_email         = mean(refr_medium == "email", na.rm = TRUE),
    andel_search        = mean(refr_medium == "search", na.rm = TRUE),
    
    # ── NYE: Tidsmæssig spredning ─────────────────────────────────────────────
    laesedage_spredning = as.numeric(max(dt) - min(dt)),
    andel_weekend       = mean(weekdays(dt) %in% c("Saturday", "Sunday")),
    andel_morgen        = mean(hour >= 6  & hour < 10),
    andel_aften         = mean(hour >= 18 & hour < 23),
    
    # ── NYE: Operativsystem ───────────────────────────────────────────────────
    andel_ios           = mean(os_family == "iOS",     na.rm = TRUE),
    andel_android       = mean(os_family == "Android", na.rm = TRUE),
    andel_windows       = mean(os_family == "Windows", na.rm = TRUE),
    
    # ── NYE: Indholdstype fra URL ─────────────────────────────────────────────
    andel_sport         = mean(str_detect(page_url_clean, "/sport/"),           na.rm = TRUE),
    andel_politik       = mean(str_detect(page_url_clean, "/politik/"),         na.rm = TRUE),
    andel_oekonomi      = mean(str_detect(page_url_clean, "/oekonomi/|/erhverv/"), na.rm = TRUE),
    andel_kultur        = mean(str_detect(page_url_clean, "/kultur/|/liv/"),    na.rm = TRUE)
  )

# ── 6. Join behavior + rens NA + behold pseudo_id ─────────────────────────────
model_data <- model_data %>%
  left_join(behavior_features, by = "pseudo_id") %>%
  mutate(
    antal_sidevisninger = replace_na(antal_sidevisninger, 0),
    antal_dage_aktiv    = replace_na(antal_dage_aktiv,    0),
    andel_restricted    = replace_na(andel_restricted,    0),
    avg_scroll_depth    = replace_na(avg_scroll_depth,    0),
    andel_mobil         = replace_na(andel_mobil,         0),
    andel_social        = replace_na(andel_social,        0),
    andel_email         = replace_na(andel_email,         0),
    andel_search        = replace_na(andel_search,        0),
    laesedage_spredning = replace_na(laesedage_spredning, 0),
    andel_weekend       = replace_na(andel_weekend,       0),
    andel_morgen        = replace_na(andel_morgen,        0),
    andel_aften         = replace_na(andel_aften,         0),
    andel_ios           = replace_na(andel_ios,           0),
    andel_android       = replace_na(andel_android,       0),
    andel_windows       = replace_na(andel_windows,       0),
    andel_sport         = replace_na(andel_sport,         0),
    andel_politik       = replace_na(andel_politik,       0),
    andel_oekonomi      = replace_na(andel_oekonomi,      0),
    andel_kultur        = replace_na(andel_kultur,        0),
    utm_medium = fct_na_value_to_level(utm_medium, "Ukendt"),
    utm_source = fct_na_value_to_level(utm_source, "Ukendt")
  )

# ── 7. Verificer ───────────────────────────────────────────────────────────────
colSums(is.na(model_data))
glimpse(model_data)

# Gem data
saveRDS(model_data, "data/model_data.rds")