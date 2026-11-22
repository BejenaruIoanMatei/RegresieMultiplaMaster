library(dplyr)
library(tidyr)
library(ggplot2)
library(psych)
library(Hmisc)
library(corrplot)

raw <- read.csv("ESS11e04_2.csv")

#selectez variabile analiza din Germania
df_de <- raw %>%
  filter(cntry == "DE") %>%
  select(
    netustm, agea, eduyrs, nwspol, wkhtot,
    wkdcorga, sclmeet, stflife, hinctnta, gndr, domicil
  )

#curatare date 
df_clean <- df_de %>%
  mutate(
    netustm  = ifelse(netustm >= 6666, NA, netustm),   # 6666 = Nu se aplică, 7777 = Refuz, 8888 = Nu știe
    agea     = ifelse(agea >= 999, NA, agea),         # 999 = Nu este disponibil
    eduyrs   = ifelse(eduyrs >= 77, NA, eduyrs),      # 77 = Refuz, 88 = Nu știe, 99 = Fără răspuns
    nwspol   = ifelse(nwspol >= 7777, NA, nwspol),    # 7777 = Refuz, 8888 = Nu știe
    wkhtot   = ifelse(wkhtot >= 666, NA, wkhtot),     # 666 = Nu se aplică, 777 = Refuz, 888 = Nu știe
    wkdcorga = ifelse(wkdcorga >= 66, NA, wkdcorga),  # 66 = Nu se aplică, 77 = Refuz, 88 = Nu știe
    sclmeet  = ifelse(sclmeet >= 77, NA, sclmeet),    # 77 = Refuz, 88 = Nu știe
    stflife  = ifelse(stflife >= 77, NA, stflife),    # 77 = Refuz, 88 = Nu știe
    hinctnta = ifelse(hinctnta >= 77, NA, hinctnta),  # 77 = Refuz, 88 = Nu știe, 99 = Fără răspuns
    gndr_fem = ifelse(gndr == 2, 1, ifelse(gndr == 1, 0, NA)),                 # 1 = Feminin, 0 = Masculin
    urban    = ifelse(domicil %in% c(1, 2, 3), 1, ifelse(domicil %in% c(4, 5), 0, NA)) # 1 = Urban, 0 = Rural
  ) %>%
  select(-gndr, -domicil) %>%   # Păstrăm doar variabilele dummy recodificate
  na.omit()

summary(df_clean)

desc_table <- psych::describe(df_clean)
print(round(desc_table, 3))

vars_cantitative <- c(
  "netustm", "agea", "eduyrs", "nwspol", "wkhtot",
  "wkdcorga", "sclmeet", "stflife", "hinctnta"
)
df_cant <- df_clean %>% select(all_of(vars_cantitative))

teste_normalitate <- lapply(df_cant, function(x) {
  sw <- shapiro.test(x)
  c(
    Shapiro_W     = round(sw$statistic, 4),
    Shapiro_p     = round(sw$p.value, 4)
  )
})

tabel_normalitate <- do.call(rbind, teste_normalitate)
print(tabel_normalitate)


rezumat_outliers <- lapply(df_cant, function(x) {
  q1  <- quantile(x, 0.25)
  q3  <- quantile(x, 0.75)
  iqr <- q3 - q1
  lim_inf <- q1 - 1.5 * iqr
  lim_sup <- q3 + 1.5 * iqr
  
  nr_inf <- sum(x < lim_inf)
  nr_sup <- sum(x > lim_sup)
  nr_tot <- nr_inf + nr_sup
  
  c(
    Q1         = round(as.numeric(q1), 2),
    Q3         = round(as.numeric(q3), 2),
    IQR        = round(as.numeric(iqr), 2),
    Lim_Inf    = round(as.numeric(lim_inf), 2),
    Lim_Sup    = round(as.numeric(lim_sup), 2),
    Out_Inf    = nr_inf,
    Out_Sup    = nr_sup,
    Total_Out  = nr_tot,
    Procent_Out = round(100 * nr_tot / length(x), 2)
  )
})

tabel_outliers <- do.call(rbind, rezumat_outliers)
print(tabel_outliers)


df_cant %>%
  pivot_longer(cols = everything(), names_to = "Variabila", values_to = "Valoare") %>%
  ggplot(aes(x = "", y = Valoare)) +
  geom_boxplot(fill = "lightblue", colour = "darkblue", outlier.colour = "red", outlier.size = 1.5) +
  facet_wrap(~ Variabila, scales = "free") +
  labs(
    title = "Diagramele box-plot ale variabilelor cantitative",
    x = "",
    y = "Valori"
  ) +
  theme_minimal()


#corelatie
cor_matrix <- cor(df_cant, method = "pearson")
print(round(cor_matrix, 3))

rcorr_res <- Hmisc::rcorr(as.matrix(df_cant), type = "pearson")
print(round(rcorr_res$r, 3)) 
print(round(rcorr_res$P, 4))

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  number.cex = 0.75,
  tl.col = "black",
  tl.srt = 45,
  diag = FALSE
)


vars_indep <- c("agea", "eduyrs", "nwspol", "wkhtot", "wkdcorga",
                "sclmeet", "stflife", "hinctnta", "gndr_fem", "urban")

cor_indep <- cor(df_clean %>% select(all_of(vars_indep)), method = "pearson")
diag(cor_indep) <- 0

cat("Corelatia absoluta maxima intre variabilele independente:",
    round(max(abs(cor_indep)), 3), "\n")

perechi_coliniare <- which(abs(cor_indep) > 0.70 & upper.tri(cor_indep), arr.ind = TRUE)
if (nrow(perechi_coliniare) == 0) {
  cat("Nu exista perechi de variabile independente cu |r| > 0.70 (nu exista coliniaritate puternica).\n")
} else {
  print(perechi_coliniare)
}

#histograme variabile cantitative 
df_cant %>%
  pivot_longer(
    cols = everything(),
    names_to = "Variabila",
    values_to = "Valoare"
  ) %>%
  ggplot(aes(x = Valoare)) +
  geom_histogram(
    bins = 25,
    fill = "steelblue",
    color = "white"
  ) +
  facet_wrap(
    ~ Variabila,
    scales = "free"
  ) +
  labs(
    title = "Histogramele variabilelor cantitative",
    x = "Valori",
    y = "Frecvență"
  ) +
  theme_minimal()

#grafic variabile dummy
ggplot(df_clean, aes(x = factor(gndr_fem))) +
  geom_bar(fill = "steelblue", width = 0.5) +
  labs(title = "Distribuția după gen", x = "Gen (0 = Masculin, 1 = Feminin)", y = "Frecvență") +
  theme_minimal()

#grafic mediu rezidenta
ggplot(df_clean, aes(x = factor(urban))) +
  geom_bar(fill = "steelblue", width = 0.5) +
  labs(title = "Distribuția după mediul de rezidență", x = "Mediu (0 = Rural, 1 = Urban)", y = "Frecvență") +
  theme_minimal()