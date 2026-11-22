# ==============================================================================
# CAPITOLUL 4: MODELARE ECONOMETRICA SI MACHINE LEARNING
# Set de date: ESS Round 11 (2023) - Germania (N = 1772)
# ==============================================================================

library(tidyverse)
library(glmnet)
library(lm.beta)

# ==============================================================================
# PASUL 1: INCARCAREA SI VERIFICAREA SETULUI DE DATE CURATAT
# ==============================================================================
df_clean <- read.csv("ESS11e04_2.csv")

if ("cntry" %in% names(df_clean)) {
  df_clean <- df_clean %>%
    filter(cntry == "DE") %>%
    select(netustm, agea, eduyrs, nwspol, wkhtot, wkdcorga, sclmeet, stflife, hinctnta, gndr, domicil) %>%
    mutate(
      netustm  = ifelse(netustm >= 6666, NA, netustm),
      agea     = ifelse(agea >= 999, NA, agea),
      eduyrs   = ifelse(eduyrs >= 77, NA, eduyrs),
      nwspol   = ifelse(nwspol >= 7777, NA, nwspol),
      wkhtot   = ifelse(wkhtot >= 666, NA, wkhtot),
      wkdcorga = ifelse(wkdcorga >= 66, NA, wkdcorga),
      sclmeet  = ifelse(sclmeet >= 77, NA, sclmeet),
      stflife  = ifelse(stflife >= 77, NA, stflife),
      hinctnta = ifelse(hinctnta >= 77, NA, hinctnta),
      gndr_fem = ifelse(gndr == 2, 1, ifelse(gndr == 1, 0, NA)),
      urban    = ifelse(domicil %in% c(1, 2, 3), 1, ifelse(domicil %in% c(4, 5), 0, NA))
    ) %>%
    select(-gndr, -domicil) %>%
    na.omit()
}

df_clean <- df_clean %>% select(-any_of(c("X", "X.1", "...1")))

# ==============================================================================
# PASUL 2: MODELUL CLASIC OLS SI METODELE SECVENTIALE (STEPWISE)
# ==============================================================================
model_nul      <- lm(netustm ~ 1, data = df_clean)
model_full     <- lm(netustm ~ ., data = df_clean)
model_backward <- step(model_full, direction = "backward", trace = 0)
model_forward  <- step(model_nul, scope = formula(model_full), direction = "forward", trace = 0)
model_step     <- step(model_full, direction = "both", trace = 1)

print(summary(model_full))
print(summary(model_step))

# Verificarea convergentei celor trei metode secventiale
cat("Convergenta Forward == Stepwise :", identical(sort(names(coef(model_forward))), sort(names(coef(model_step)))), "\n")
cat("Convergenta Backward == Stepwise:", identical(sort(names(coef(model_backward))), sort(names(coef(model_step)))), "\n")

# ==============================================================================
# PASUL 3: MODELE DE REGULARIZARE / ML (RIDGE, LASSO, ELASTIC NET)
# ==============================================================================
X <- model.matrix(netustm ~ ., data = df_clean)[, -1]
y <- df_clean$netustm

set.seed(123); cv_ridge <- cv.glmnet(X, y, alpha = 0, nfolds = 10)
set.seed(123); cv_lasso <- cv.glmnet(X, y, alpha = 1, nfolds = 10)
set.seed(123); cv_enet  <- cv.glmnet(X, y, alpha = 0.5, nfolds = 10)

coef_ridge     <- as.matrix(coef(cv_ridge, s = "lambda.min"))
coef_lasso     <- as.matrix(coef(cv_lasso, s = "lambda.min"))
coef_lasso_1se <- as.matrix(coef(cv_lasso, s = "lambda.1se"))
coef_enet      <- as.matrix(coef(cv_enet,  s = "lambda.min"))

png("grafice_validare_ML_Cap4.png", width = 1200, height = 450, res = 120)
par(mfrow = c(1, 3))
plot(cv_ridge); title("Validare incrucisata: Ridge", line = 2.5)
plot(cv_lasso); title("Validare incrucisata: Lasso", line = 2.5)
plot(cv_enet);  title("Validare incrucisata: Elastic Net", line = 2.5)
dev.off()

# ==============================================================================
# PASUL 4: INTERPRETAREA MODELULUI FINAL ALES SI EXPORTUL TABELELOR
# ==============================================================================
calc_rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
calc_mae  <- function(actual, pred) mean(abs(actual - pred))
calc_r2   <- function(actual, pred) 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)

pred_ols_full <- predict(model_full, newdata = df_clean)
pred_step     <- predict(model_step, newdata = df_clean)
pred_ridge    <- as.numeric(predict(cv_ridge, newx = X, s = "lambda.min"))
pred_lasso    <- as.numeric(predict(cv_lasso, newx = X, s = "lambda.min"))
pred_enet     <- as.numeric(predict(cv_enet,  newx = X, s = "lambda.min"))

# --- TABEL 1: Comparatie coeficienti ---
toti_termenii <- names(coef(model_full))
coef_step_aliniat <- setNames(rep(0, length(toti_termenii)), toti_termenii)
coef_step_aliniat[names(coef(model_step))] <- coef(model_step)

tabel1_coeficienti <- data.frame(
  Variabila      = toti_termenii,
  OLS_Complet    = round(as.numeric(coef(model_full)), 4),
  Stepwise       = round(as.numeric(coef_step_aliniat), 4),
  Ridge_min      = round(as.numeric(coef_ridge), 4),
  Lasso_min      = round(as.numeric(coef_lasso), 4),
  Lasso_1se      = round(as.numeric(coef_lasso_1se), 4),
  ElasticNet_min = round(as.numeric(coef_enet), 4)
)

# --- TABEL 2: Indicatori de performanta (inclusiv RMSE din 10-fold CV) ---
tabel2_performanta <- data.frame(
  Model = c("1. OLS Complet", "2. Secvential (Stepwise)", "3. Regresie Ridge (alpha=0)",
            "4. Regresie Lasso (alpha=1)", "5. Elastic Net (alpha=0.5)"),
  Nr_Variabile_Active = c(length(coef(model_full)) - 1, length(coef(model_step)) - 1,
                          sum(coef_ridge[-1] != 0), sum(coef_lasso[-1] != 0), sum(coef_enet[-1] != 0)),
  Lambda_Optim = c("-", "-", round(cv_ridge$lambda.min, 4), round(cv_lasso$lambda.min, 4), round(cv_enet$lambda.min, 4)),
  RMSE_Train = round(c(calc_rmse(y, pred_ols_full), calc_rmse(y, pred_step), calc_rmse(y, pred_ridge),
                       calc_rmse(y, pred_lasso), calc_rmse(y, pred_enet)), 3),
  RMSE_CV    = c("-", "-", round(sqrt(min(cv_ridge$cvm)), 3),
                 round(sqrt(min(cv_lasso$cvm)), 3), round(sqrt(min(cv_enet$cvm)), 3)),
  MAE  = round(c(calc_mae(y, pred_ols_full), calc_mae(y, pred_step), calc_mae(y, pred_ridge),
                 calc_mae(y, pred_lasso), calc_mae(y, pred_enet)), 3),
  R2   = round(c(calc_r2(y, pred_ols_full), calc_r2(y, pred_step), calc_r2(y, pred_ridge),
                 calc_r2(y, pred_lasso), calc_r2(y, pred_enet)), 4),
  AIC  = c(round(AIC(model_full), 2), round(AIC(model_step), 2), "-", "-", "-"),
  BIC  = c(round(BIC(model_full), 2), round(BIC(model_step), 2), "-", "-", "-")
)

# --- Indicatorii de corelatie si Testarea globala (Fisher F) ---
s_step   <- summary(model_step)
R_mult   <- sqrt(s_step$r.squared)
f_calc   <- as.numeric(s_step$fstatistic[1])
df1      <- as.numeric(s_step$fstatistic[2])
df2      <- as.numeric(s_step$fstatistic[3])
f_critic <- qf(0.95, df1, df2)
t_critic <- qt(0.975, df2)

cat("\n==================== INDICATORI DE CORELATIE SI TESTUL FISHER ====================\n")
cat("Raportul de corelatie multipla (R)  :", round(R_mult, 4), "\n")
cat("Coeficientul de determinatie (R^2)  :", round(s_step$r.squared, 4), "\n")
cat("R^2 ajustat                         :", round(s_step$adj.r.squared, 4), "\n")
cat("Statistica F calculata              :", round(f_calc, 4), "(df1 =", df1, ", df2 =", df2, ")\n")
cat("Valoare critica F (alpha = 0.05)    :", round(f_critic, 4), "\n")
cat("Valoare critica t (alpha = 0.05)    :", round(t_critic, 4), "\n")

# --- Modelul standardizat explicit + Corelatii partiale (NA la Intercept) ---
model_std_explicit <- lm(
  scale(netustm) ~ scale(agea) + scale(eduyrs) + scale(nwspol) + scale(wkhtot) +
    scale(wkdcorga) + scale(stflife) + scale(hinctnta) + scale(urban),
  data = df_clean
)
print(summary(model_std_explicit))

sum_step <- s_step$coefficients
ic_step  <- confint(model_step, level = 0.95)
beta_std <- coef(lm.beta(model_step))
t_vals   <- sum_step[, "t value"]
r_part   <- ifelse(rownames(sum_step) == "(Intercept)", NA, round(t_vals / sqrt(t_vals^2 + df2), 4))

tabel3_model_final <- data.frame(
  Variabila         = rownames(sum_step),
  Coef_Nestandard   = round(sum_step[, "Estimate"], 4),
  Eroare_Standard   = round(sum_step[, "Std. Error"], 4),
  IC_95_Min         = round(ic_step[, 1], 4),
  IC_95_Max         = round(ic_step[, 2], 4),
  Statistica_t      = round(t_vals, 3),
  P_value           = round(sum_step[, "Pr(>|t|)"], 5),
  Semnificativ_5pct = ifelse(sum_step[, "Pr(>|t|)"] < 0.05, "Da", "Nu"),
  Corelatie_Partial = r_part,
  Coef_Standardizat = round(as.numeric(beta_std), 4),
  Importanta_Abs    = round(abs(as.numeric(beta_std)), 4)
) %>%
  arrange(desc(Importanta_Abs))

# --- Determinarea valorii lui Y pentru profiluri date ---
profiluri <- data.frame(
  Profil   = c("Profil 1: Tanar urban, educat, autonomie mare", "Profil 2: Varstnic rural, educatie medie, autonomie mica"),
  agea     = c(28, 65),
  eduyrs   = c(17, 12),
  nwspol   = c(60, 45),
  wkhtot   = c(42, 35),
  wkdcorga = c(9, 4),
  stflife  = c(8, 7),
  hinctnta = c(8, 4),
  urban    = c(1, 0)
)
pred_profiluri <- cbind(
  Profil = profiluri$Profil,
  round(predict(model_step, newdata = profiluri, interval = "confidence"), 2)
)

# --- Determinarea variatiei lui Y (Delta Y) la variatii ale lui X ---
b <- coef(model_step)
tabel4_variatii <- data.frame(
  Scenariu_Variatie_X = c(
    "Cresterea varstei cu +10 ani",
    "Cresterea autonomiei muncii cu +2 puncte (scala 0-10)",
    "Cresterea duratei muncii cu +8 ore/saptamana",
    "Cresterea educatiei cu +4 ani (studii universitare)",
    "Cresterea satisfactiei fata de viata cu +2 puncte",
    "Cresterea timpului pentru stiri politice cu +30 min/zi",
    "Schimbarea mediului din rural (0) in urban (1)"
  ),
  Variabila  = c("agea", "wkdcorga", "wkhtot", "eduyrs", "stflife", "nwspol", "urban"),
  Delta_X    = c(10, 2, 8, 4, 2, 30, 1),
  Coef_Beta  = round(as.numeric(b[c("agea", "wkdcorga", "wkhtot", "eduyrs", "stflife", "nwspol", "urban")]), 4)
) %>%
  mutate(Variatie_Estimata_Y_min = round(Coef_Beta * Delta_X, 2))

print(tabel1_coeficienti)
print(tabel2_performanta)
print(tabel3_model_final)
print(pred_profiluri)
print(tabel4_variatii)

write.csv(tabel1_coeficienti, "Tabel1_Comparatie_Coeficienti_Cap4.csv", row.names = FALSE)
write.csv(tabel2_performanta, "Tabel2_Performanta_Modele_Cap4.csv", row.names = FALSE)
write.csv(tabel3_model_final, "Tabel3_Model_Final_Standardizat_Cap4.csv", row.names = FALSE)
write.csv(tabel4_variatii,    "Tabel4_Scenarii_Variatie_DeltaY_Cap4.csv", row.names = FALSE)