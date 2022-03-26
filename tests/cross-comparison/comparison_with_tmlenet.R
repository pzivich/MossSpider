##############################################################
# Network-TMLE code to verify against
##############################################################

# library(devtools)
# devtools::install_github('osofr/simcausal', build_vignettes = FALSE)
# devtools::install_github('osofr/tmlenet', build_vignettes = FALSE)
library(tmlenet)

Kmax <- 2 # max number of friends in the network
n <- 1000 # number of obs

data(df_netKmax2)
write.csv(df_netKmax2[c("IDs", "W1", "A", "Y", "Net_str")], file="tmlenet_r_data.csv")

sW <- def_sW(sum.netW1 = sum(W1[[1:Kmax]]), replaceNAw0=TRUE)

sA <- def_sA(A = A) +
  def_sA(netA = A[[1:Kmax]], replaceNAw0=TRUE) +
  def_sA(sum.netA = sum(A[[1:Kmax]]), replaceNAw0=TRUE)

summaries = eval.summaries(data = df_netKmax2, Kmax = Kmax,
               sW = sW, sA = sA, IDnode = "IDs", NETIDnode = "Net_str")
# head(summaries$sA.matrix)
df_for_sas = summaries$DatNet.ObsP0$dat.sWsA
# write.csv(df_for_sas, file='tmle_r_data_processed.csv')

head(summaries$DatNet.ObsP0$mat.sVar, 20)
d <- summaries$DatNet.ObsP0$mat.sVar
glm("Y ~ A + sum.netA + W1 + sum.netW1", family="binomial", data=d)
glm("A ~ W1 + sum.netW1", family="binomial", data=d)

d$A2 = rbinom(nrow(df_netKmax2), 1, 0.35)
glm("A2 ~ W1 + sum.netW1", family="binomial", data=d)


write.csv(d, file="/home/pzivich/Desktop/R_gen_data.csv")

options(tmlenet.verbose = TRUE, useglm=T)
res_K2_1a <- tmlenet(data = df_netKmax2, Kmax = Kmax,
                     # Anodes = "A", # f_gstar1 = f.A_0,
                     # intervene1.sA = def_new_sA(A = 0),
                     intervene1.sA = def_new_sA(A=rbinom(nrow(df_netKmax2), 1, 0.35)),
                     sW = sW, sA = sA,
                     Qform = "Y ~ A + sum.netA + W1 + sum.netW1",
                     hform.g0 = "A + netA ~ W1 + sum.netW1",
                     IDnode = "IDs", NETIDnode = "Net_str",
                     optPars=list(n_MCsims=1000))

res_K2_1a$EY_gstar1$estimates  # psi: 0.
res_K2_1a$EY_gstar1$condW.IC.vars  # Var(psi): 0.
res_K2_1a$EY_gstar1$condW.CIs  # psi CL:  
# res_K2_1a$EY_gstar1$condW.indepQ.IC.vars # Var(psi): 0.000536523
# res_K2_1a$EY_gstar1$condW.indepQ.CIs  # psi CL: 0.4607124  0.5515096

res_K2_1b <- tmlenet(data = df_netKmax2, Kmax = Kmax,
                     # Anodes = "A", # f_gstar1 = f.A_0,
                     # intervene1.sA = def_new_sA(A = 0),
                     intervene1.sA = def_new_sA(A=rbinom(nrow(df_netKmax2), 1, 0.35)),
                     sW = sW, sA = sA,
                     Qform = "Y ~ A + sum.netA + W1 + sum.netW1",
                     hform.g0 = "A + sum.netA ~ W1 + sum.netW1",
                     IDnode = "IDs", NETIDnode = "Net_str")

res_K2_1b$EY_gstar1$estimates  # psi: 0.
res_K2_1b$EY_gstar1$condW.IC.vars  # Var(psi): 0.
res_K2_1b$EY_gstar1$condW.CIs  # psi CL:  

res_K2_1a$EY_gstar1$h_gstar_SummariesModel$predvars
res_K2_1a$EY_gstar1$h_gstar_SummariesModel



options(tmlenet.verbose = FALSE, useglm=T)
start <- Sys.time()
res_K2_1a <- tmlenet(data = df_netKmax2, Kmax = Kmax,
                     intervene1.sA = def_new_sA(A=rbinom(nrow(df_netKmax2), 1, 0.35)),
                     f_gstar1 = def_new_sA(A=rbinom(nrow(df_netKmax2), 1, 0.35)),
                     sW = sW, sA = sA,
                     Qform = "Y ~ A + sum.netA + W1 + sum.netW1",
                     hform.g0 = "A + netA ~ W1 + sum.netW1",
                     IDnode = "IDs", NETIDnode = "Net_str",
                     optPars=list(n_MCsims=5))
print(Sys.time() - start)

start <- Sys.time()
res_K2_1a <- tmlenet(data = df_netKmax2, Kmax = Kmax,
                     intervene1.sA = def_new_sA(A=rbinom(nrow(df_netKmax2), 1, 0.35)),
                     sW = sW, sA = sA,
                     Qform = "Y ~ A + sum.netA + W1 + sum.netW1",
                     hform.g0 = "A + netA ~ W1 + sum.netW1",
                     IDnode = "IDs", NETIDnode = "Net_str",
                     optPars=list(n_MCsims=1000000))
print(Sys.time() - start)


total_results = c()
policies = c(0.7, 0.8, 0.9)
for (j in policies){
  results = c()
  for (i in 1:2000){
    res_K2_1a <- tmlenet(data = df_netKmax2, Kmax = Kmax,
                         intervene1.sA = def_new_sA(A=rbinom(nrow(df_netKmax2), 1, j)),
                         sW = sW, sA = sA,
                         Qform = "Y ~ A + sum.netA + W1 + sum.netW1",
                         hform.g0 = "A + netA ~ W1 + sum.netW1",
                         IDnode = "IDs", NETIDnode = "Net_str",
                         optPars=list(n_MCsims=1))
    results <- c(results, res_K2_1a$EY_gstar1$estimates[1])
  }
  total_results = c(total_results, mean(results))
}
print(total_results)


res_K2_1a$EY_gstar1$condW.CIs  # psi CL:  

res_K2_1a <- tmlenet(data = df_netKmax2, Kmax = Kmax,
                     intervene1.sA = def_new_sA(A=rbinom(nrow(df_netKmax2), 1, 0.35)),
                     sW = sW, sA = sA,
                     Qform = "Y ~ A + sum.netA + W1 + sum.netW1",
                     hform.g0 = "A + netA ~ W1 + sum.netW1",
                     IDnode = "IDs", NETIDnode = "Net_str",
                     optPars=list(n_MCsims=1000000))
res_K2_1a$EY_gstar1$estimates  # psi: 0.
res_K2_1a$EY_gstar1$condW.CIs  # psi CL:  
