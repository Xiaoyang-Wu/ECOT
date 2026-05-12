library(MASS)
library(foreach)
library(randomForest)
library(doParallel)
library(e1071)
library(dplyr)

source("functions.R")


dataori <- read.csv('mammography.csv', header = T)
names(dataori) <- c('X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y')
dataori$Y <- ifelse(dataori$Y==dataori$Y[1], '-1', '1')
data_clean <- dataori[dataori$Y=='-1', ]
data_out <- dataori[dataori$Y=='1', ]



n <- 380
n1 <- 20
nfull <- n + n1
pi0l <- n/nfull
m <- 1000
pi0 <- 0.95
pi1 <- 0.05
alpha <- 0.1
d <- 6


algoarray <- data.frame(matrix(NA, nrow = 6, ncol = 2))
names(algoarray) <- c('algotype', 'param')
algoarray$algotype <- rep(c('RF', 'SVM'), each = 3)
algoarray$param <- c(100, 300, 500, 0.1, 0.2, 0.4)


nr <- 500
cl <- makeCluster(10)
registerDoParallel(cl)
Result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "isotree", "e1071", "randomForest"), .errorhandling = "remove")%dopar% {
  
  data <- data.frame()
  
  data_clean_use <- data_clean[sample(1:nrow(data_clean), nrow(data_clean)),]
  data_out_use <- data_out[sample(1:nrow(data_out), nrow(data_out)),]
  
  X0 <- data_clean_use[1:n, 1:d]
  X_test <- rbind(data_out_use[1:(pi1*m), 1:d], data_clean_use[(n+1):(n+pi0*m), 1:d])
  outlier <- 1:(pi1*m)
  X1 <- data_out_use[(pi1*m+1):(pi1*m+n1), 1:d]
    
  X0_train <- X0[1:(n/2),]
  X0_cal <- X0[(n/2+1):n,]
  X1_train <- X1[1:(n1/2),]
  X1_cal <- X1[(n1/2+1):n1,]
  
  m_clean <- ceiling(m/2)
  
  
  ocs_model_list <- list()
  pvalB_mat <- matrix(NA, nrow = m, ncol = nrow(algoarray))
  VBcal_mat <- matrix(NA, nrow = (n+n1)/2, ncol = nrow(algoarray))
  VBtest_mat <- matrix(NA, nrow = m, ncol = nrow(algoarray))
  t_before <- proc.time()
  for (i in 1:nrow(algoarray)) {
    if (algoarray$algotype[i]=='RF'){
      ocs_temp_model <- randomForest(y~., data = data.frame(x = rbind(X0_train, X1_train), y = factor(c(rep(0, n/2), rep(1, n1/2)))), ntree = algoarray$param[i])
      ocs_model_list[[i]] <- list(model = ocs_temp_model, type = algoarray$algotype[i], param = algoarray$param[i])
      VBcal_mat[, i] <- predict(ocs_temp_model, data.frame(x = rbind(X0_cal, X1_cal)), type = 'prob')[, 2] - c(rep(0, n/2), rep(1, n1/2))*1000
      VBtest_mat[, i] <- predict(ocs_temp_model, data.frame(x = X_test), type = 'prob')[, 2]
      pvalB_mat[, i] <- sapply(VBtest_mat[, i], function(x){(sum(x<=VBcal_mat[, i])+1)/(length(VBcal_mat[, i]+1))})
    }
    if (algoarray$algotype[i]=='SVM'){
      ocs_temp_model <- svm(y~., data = data.frame(x = rbind(X0_train, X1_train), y = factor(c(rep(0, n/2), rep(1, n1/2)))), probability = T, nu = algoarray$param[i])
      ocs_model_list[[i]] <- list(model = ocs_temp_model, type = algoarray$algotype[i], param = algoarray$param[i])
      VBcal_mat[, i] <- attr(predict(ocs_temp_model, data.frame(x = rbind(X0_cal, X1_cal)), probability = T), 'probabilities')[, 2] - c(rep(0, n/2), rep(1, n1/2))*1000
      VBtest_mat[, i] <- attr(predict(ocs_temp_model, data.frame(x = X_test), probability = T), 'probabilities')[, 2]
      pvalB_mat[, i] <- sapply(VBtest_mat[, i], function(x){(sum(x<=VBcal_mat[, i])+1)/(length(VBcal_mat[, i]+1))})
    }
  }
  t_after <- proc.time()
  t_ocs <- (t_after - t_before)['elapsed']
  
  
  t_ecot <- 0
  t_ecot_clean <- 0
  ecot_clean_model_list <- list()
  pval_clean_mat <- matrix(NA, nrow = m, ncol = nrow(algoarray))
  Vncal_clean_mat <- matrix(NA, nrow = n, ncol = nrow(algoarray))
  Vntest_clean_mat <- matrix(NA, nrow = m, ncol = nrow(algoarray))
  for (i in 1:nrow(algoarray)) {
    if (algoarray$algotype[i]=='RF'){
      t_before <- proc.time()
      ecot_temp_model <- randomForest(y~., data = data.frame(x = rbind(X0, X_test, X1), y = factor(c(rep(0, n+m), rep(1, n1)))), ntree = algoarray$param[i])
      t_after <- proc.time()
      t_ecot <- t_ecot + (t_after - t_before)['elapsed']
      
      
      t_before <- proc.time()
      V_idf_temp <- predict(ecot_temp_model, data.frame(x = rbind(X0, X_test)), type = 'prob')[, 2]
      if(sum(V_idf_temp==sort(V_idf_temp)[n+m_clean])==1){
        X0_clean_temp <- rbind(X0, X_test)[which(V_idf_temp<=sort(V_idf_temp)[n+m_clean]), ]
      }else{
        X0_clean_temp <- rbind(X0, X_test)[c(which(V_idf_temp<sort(V_idf_temp)[n+m_clean]), sample(which(V_idf_temp==sort(V_idf_temp)[n+m_clean]), n+m_clean-length(which(V_idf_temp<sort(V_idf_temp)[n+m_clean])))), ]
      }
      ecot_temp_model_clean <- randomForest(y~., data = data.frame(x = rbind(X0_clean_temp, X1), y = factor(c(rep(0, n+m_clean), rep(1, n1)))), ntree = algoarray$param[i])
      ecot_clean_model_list[[i]] <- list(model = ecot_temp_model_clean, type = algoarray$algotype[i], param = algoarray$param[i])
      Vncal_clean_mat[, i] <- predict(ecot_temp_model_clean, data.frame(x = X0), type = 'prob')[, 2]
      Vntest_clean_mat[ ,i] <- predict(ecot_temp_model_clean, data.frame(x = X_test), type = 'prob')[, 2]
      pval_clean_mat[, i] <- sapply(Vntest_clean_mat[ ,i], function(x){(sum(x<=Vncal_clean_mat[, i])+1)/(length(Vncal_clean_mat[, i])+1)})
      t_after <- proc.time()
      t_ecot_clean <- t_ecot_clean + (t_after - t_before)['elapsed']
    }
    if (algoarray$algotype[i]=='SVM'){
      t_before <- proc.time()
      ecot_temp_model <- svm(y~., data = data.frame(x = rbind(X0, X_test, X1), y = factor(c(rep(0, n+m), rep(1, n1)))), probability = T, nu = algoarray$param[i])
      t_after <- proc.time()
      t_ecot_clean <- t_ecot_clean + (t_after - t_before)['elapsed']
      
      
      t_before <- proc.time()
      V_idf_temp <- attr(predict(ecot_temp_model, data.frame(x = rbind(X0, X_test)), probability = T), 'probabilities')[, 2]
      if(sum(V_idf_temp==sort(V_idf_temp)[n+m_clean])==1){
        X0_clean_temp <- rbind(X0, X_test)[which(V_idf_temp<=sort(V_idf_temp)[n+m_clean]), ]
      }else{
        X0_clean_temp <- rbind(X0, X_test)[c(which(V_idf_temp<sort(V_idf_temp)[n+m_clean]), sample(which(V_idf_temp==sort(V_idf_temp)[n+m_clean]), n+m_clean-length(which(V_idf_temp<sort(V_idf_temp)[n+m_clean])))), ]
      }
      ecot_temp_model_clean <- svm(y~., data = data.frame(x = rbind(X0_clean_temp, X1), y = factor(c(rep(0, n+m_clean), rep(1, n1)))), probability = T, nu = algoarray$param[i])
      ecot_clean_model_list[[i]] <- list(model = ecot_temp_model_clean, type = algoarray$algotype[i], param = algoarray$param[i])
      Vncal_clean_mat[, i] <- attr(predict(ecot_temp_model_clean, data.frame(x = X0), probability = T), 'probabilities')[, 2]
      Vntest_clean_mat[, i] <- attr(predict(ecot_temp_model_clean, data.frame(x = X_test), probability = T), 'probabilities')[, 2]
      pval_clean_mat[, i] <- sapply(Vntest_clean_mat[ ,i], function(x){(sum(x<=Vncal_clean_mat[, i])+1)/(length(Vncal_clean_mat[, i])+1)})
      t_after <- proc.time()
      t_ecot_clean <- t_ecot_clean + (t_after - t_before)['elapsed']
    }
  }
  
  
  model_RF_B <- randomForest(y~., data = data.frame(x = rbind(X0_train, X1_train), y = factor(c(rep(0, n/2), rep(1, n1/2)))), ntree = 500)
  model_RF_new <- randomForest(y~., data = data.frame(x = rbind(X0, X_test, X1), y = factor(c(rep(0, n+m), rep(1, n1)))), ntree = 500)
  
  
  V_idf <- predict(model_RF_new, data.frame(x = rbind(X0, X_test)), type = 'prob')[, 2]
  if(sum(V_idf==sort(V_idf)[n+m_clean])==1){
    X0_clean <- rbind(X0, X_test)[which(V_idf<=sort(V_idf)[n+m_clean]), ]
  }else{
    X0_clean <- rbind(X0, X_test)[c(which(V_idf<sort(V_idf)[n+m_clean]), sample(which(V_idf==sort(V_idf)[n+m_clean]), n+m_clean-length(which(V_idf<sort(V_idf)[n+m_clean])))), ]
  }
  model_RF_new_clean <- randomForest(y~., data = data.frame(x = rbind(X0_clean, X1), y = factor(c(rep(0, n+m_clean), rep(1, n1)))), ntree = 500)
  
  
  VB_cal <- predict(model_RF_B, data.frame(x = rbind(X0_cal, X1_cal)), type = 'prob')[, 2] - c(rep(0, n/2), rep(1, n1/2))*100
  VB_test <- predict(model_RF_B, data.frame(x = X_test), type = 'prob')[, 2]
  Vn_cal <- predict(model_RF_new, data.frame(x = X0), type = 'prob')[, 2]
  Vn_test <- predict(model_RF_new, data.frame(x = X_test), type = 'prob')[, 2]
  Vn_cal_clean <- predict(model_RF_new_clean, data.frame(x = X0), type = 'prob')[, 2]
  Vn_test_clean <- predict(model_RF_new_clean, data.frame(x = X_test), type = 'prob')[, 2]
  
  
  ###---OCS model selection---###
  rejnum_B_sel <- rep(NA, m)
  pval_B_sel <- rep(NA, m)
  ind_B_vec <- rep(NA, m)
  t_before <- proc.time()
  for (j in 1:m){
    rejnum_B_each <- rep(NA, nrow(algoarray))
    
    for (k in 1:nrow(algoarray)) {
      VB_cal_sel <- c(VBcal_mat[, k], VBtest_mat[j, k])
      VB_test_sel <- VBtest_mat[-j, k]
      
      pval_B_sel_aux <- sapply(VB_test_sel, function(x){sum(x<=VB_cal_sel)/length(VB_cal_sel)})
      if(j==1){
        pval_B_sel_aux <- c(0, pval_B_sel_aux)
      }else if(j==m){
        pval_B_sel_aux <- c(pval_B_sel_aux, 0)
      }else {
        pval_B_sel_aux <- c(pval_B_sel_aux[1:(j-1)], 0, pval_B_sel_aux[j:(m-1)])
      }
      rejnum_B_each[k] <- length(BH(pval_B_sel_aux, alpha))
    }
    
    if(sum(rejnum_B_each==max(rejnum_B_each))==1){
      ind_B <- which(rejnum_B_each==max(rejnum_B_each))
    }else {
      ind_B <- sample(which(rejnum_B_each==max(rejnum_B_each)), 1)
    }
    
    ind_B_vec[j] <- ind_B
    rejnum_B_sel[j] <- max(rejnum_B_each)
    pval_B_sel[j] <- pvalB_mat[j, ind_B]
  }
  rej_B_sel_init <- which(pval_B_sel<=alpha*rejnum_B_sel/m)
  if(length(rej_B_sel_init)>=max(rejnum_B_sel)){
    rej_B_sel <- rej_B_sel_init
  }else {
    rej_B_sel <- rej_B_sel_init[BH(runif(length(rej_B_sel_init))*rejnum_B_sel[rej_B_sel_init]/length(rej_B_sel_init), 1)]
  }
  rej_B_sel <- BH(pval_B_sel, alpha)
  t_after <- proc.time()
  power_B_sel <- sum(rej_B_sel%in%outlier)/length(outlier)
  FDP_B_sel <- sum(!rej_B_sel%in%outlier)/max(length(rej_B_sel), 1)
  t_ocs <- t_ocs + (t_after - t_before)["elapsed"]
  data <- rbind(data, data.frame(t = t_ocs, FDP = FDP_B_sel, POWER = power_B_sel, method = 'OptCS-MSel', n = nfull, alpha = alpha))
  
  
  ###---ECOT method selection---###
  rejnum_sel_clean <- rep(NA, m)
  pval_sel_clean <- rep(NA, m)
  ind_vec_clean <- rep(NA, m)
  t_before <- proc.time()
  for (j in 1:m){
    rejnum_each_clean <- rep(NA, nrow(algoarray))
    rejnum_each_mod_clean <- rep(NA, nrow(algoarray))
    
    for (k in 1:nrow(algoarray)) {
      Vn_clean_cal_sel <- c(Vncal_clean_mat[, k], Vntest_clean_mat[j, k])
      rem_bi_clean <- which.max(Vn_clean_cal_sel)
      Vn_clean_cal_sel <- Vn_clean_cal_sel[-rem_bi_clean]
      Vn_clean_test_sel <- Vntest_clean_mat[-j, k]
      
      pval_clean_new_sel <- sapply(Vn_clean_test_sel, function(x){(sum(x<=Vn_clean_cal_sel)+1)/(length(Vn_clean_cal_sel)+1)})
      if(j==1){
        pval_clean_new_sel <- c(0, pval_clean_new_sel)
      }else if(j==m){
        pval_clean_new_sel <- c(pval_clean_new_sel, 0)
      }else {
        pval_clean_new_sel <- c(pval_clean_new_sel[1:(j-1)], 0, pval_clean_new_sel[j:(m-1)])
      }
      pval_clean_new_sel_mod <- sapply(pval_clean_new_sel, function(x){x-1/(length(Vn_clean_cal_sel)+1)})
      pval_clean_new_sel <- pval_clean_new_sel*pi0l
      pval_clean_new_sel_mod <- pval_clean_new_sel_mod*pi0l
      pval_clean_new_sel_mod[j] <- 0
      rejnum_each_clean[k] <- length(BH(pval_clean_new_sel, alpha))
      rejnum_each_mod_clean[k] <- length(BH(pval_clean_new_sel_mod, alpha))
    }
    
    if(sum(rejnum_each_clean==1)<nrow(algoarray)){
      if(sum(rejnum_each_clean==max(rejnum_each_clean))==1){
        ind_clean <- which(rejnum_each_clean==max(rejnum_each_clean))
        rejnum_sel_clean[j] <- max(rejnum_each_clean)
      }else {
        if(sum(rejnum_each_mod_clean==max(rejnum_each_mod_clean))==1){
          ind_clean <- which(rejnum_each_mod_clean==max(rejnum_each_mod_clean))
        }else {
          ind_clean <- sample(which(rejnum_each_mod_clean==max(rejnum_each_mod_clean)), 1)
        }
        rejnum_sel_clean[j] <- max(rejnum_each_mod_clean)
      }
    }else {
      if(sum(rejnum_each_mod_clean==max(rejnum_each_mod_clean))==1){
        ind_clean <- which(rejnum_each_mod_clean==max(rejnum_each_mod_clean))
      }else {
        ind_clean <- sample(which(rejnum_each_mod_clean==max(rejnum_each_mod_clean)), 1)
      }
      rejnum_sel_clean[j] <- max(rejnum_each_mod_clean)
    }
    
    ind_vec_clean[j] <- ind_clean
    pval_sel_clean[j] <- pval_clean_mat[j, ind_clean]*pi0l
  }
  rej_sel_clean_init <- which(pval_sel_clean<=alpha*rejnum_sel_clean/m)
  if(length(rej_sel_clean_init)>=max(rejnum_sel_clean)){
    rej_sel_clean <- rej_sel_clean_init
  }else {
    rej_sel_clean <- rej_sel_clean_init[BH(runif(length(rej_sel_clean_init))*rejnum_sel_clean[rej_sel_clean_init]/length(rej_sel_clean_init), 1)]
  }
  rej_sel_clean <- BH(pval_sel_clean, alpha)
  t_after <- proc.time()
  t_ecot_clean <- t_ecot + t_ecot_clean + (t_after - t_before)["elapsed"]
  power_sel_clean <- sum(rej_sel_clean%in%outlier)/length(outlier)
  FDP_sel_clean <- sum(!rej_sel_clean%in%outlier)/max(length(rej_sel_clean), 1)
  data <- rbind(data, data.frame(t = t_ecot_clean, FDP = FDP_sel_clean, POWER = power_sel_clean, method = 'ECOT-as', n = nfull, alpha = alpha))
  
  
  ###OCS-full###
  pval_ocsfull <- rep(NA, m)
  ind_full_vec <- rep(NA, m)
  Vfullcal_mat <- matrix(NA, nrow = nfull, ncol = nrow(algoarray))
  Vfulltest_mat <- matrix(NA, nrow = m, ncol = nrow(algoarray))
  t_before <- proc.time()
  for (j in 1:(nfull+m)) {
    for (i in 1:nrow(algoarray)) {
      if (algoarray$algotype[i]=='RF'){
        ocsfull_temp_model <- randomForest(y~., data = data.frame(x = rbind(X0, X1, X_test)[-j, ], y = factor(c(rep(0, n), rep(1, n1), rep(0, m))[-j])), ntree = algoarray$param[i])
        if (j<=nfull) {
          Vfullcal_mat[j, i] <- predict(ocsfull_temp_model, data.frame(x = rbind(X0, X1, X_test)[j, ]), type = 'prob')[2] - c(rep(0, n), rep(1, n1), rep(0, m))[j]*1000
        } else {
          Vfulltest_mat[j-nfull, i] <- predict(ocsfull_temp_model, data.frame(x = rbind(X0, X1, X_test)[j, ]), type = 'prob')[2]
        }
      }
      if (algoarray$algotype[i]=='SVM'){
        ocsfull_temp_model <- svm(y~., data = data.frame(x = rbind(X0, X1, X_test)[-j, ], y = factor(c(rep(0, n), rep(1, n1), rep(0, m))[-j])), probability = T, nu = algoarray$param[i])
        if (j<=nfull) {
          Vfullcal_mat[j, i] <- attr(predict(ocsfull_temp_model, data.frame(x = rbind(X0, X1, X_test)[j, ]), probability = T), 'probabilities')[2] - c(rep(0, n), rep(1, n1), rep(0, m))[j]*1000
        } else {
          Vfulltest_mat[j-nfull, i] <- attr(predict(ocsfull_temp_model, data.frame(x = rbind(X0, X1, X_test)[j, ]), probability = T), 'probabilities')[2]
        }
      }
    }
  }
  for (j in 1:m) {
    rejnum_full_each <- rep(NA, nrow(algoarray))
    
    for (k in 1:nrow(algoarray)) {
      Vfull_cal_sel <- c(Vfullcal_mat[, k], Vfulltest_mat[j, k])
      Vfull_test_sel <- Vfulltest_mat[-j, k]
      pval_full_sel_aux <- sapply(Vfull_test_sel, function(x){sum(x<=Vfull_cal_sel)/length(Vfull_cal_sel)})
      if(j==1){
        pval_full_sel_aux <- c(0, pval_full_sel_aux)
      }else if(j==m){
        pval_full_sel_aux <- c(pval_full_sel_aux, 0)
      }else {
        pval_full_sel_aux <- c(pval_full_sel_aux[1:(j-1)], 0, pval_full_sel_aux[j:(m-1)])
      }
      rejnum_full_each[k] <- length(BH(pval_full_sel_aux, alpha))
    }
    if(sum(rejnum_full_each==max(rejnum_full_each))==1){
      ind_full <- which(rejnum_full_each==max(rejnum_full_each))
    }else {
      ind_full <- sample(which(rejnum_full_each==max(rejnum_full_each)), 1)
    }
    
    ind_full_vec[j] <- ind_full
    pval_ocsfull[j] <- (sum(Vfulltest_mat[j, ind_full]<=Vfullcal_mat[, ind_full])+1)/(length(Vfullcal_mat[, ind_full])+1)
  }
  rej_ocsfull <- BH(pval_ocsfull, alpha)
  t_after <- proc.time()
  power_ocsfull <- sum(rej_ocsfull%in%outlier)/length(outlier)
  FDP_ocsfull <- sum(!rej_ocsfull%in%outlier)/max(length(rej_ocsfull), 1)
  t_ocs_full <- (t_after - t_before)["elapsed"]
  data <- rbind(data, data.frame(t = t_ocs_full, FDP = FDP_ocsfull, POWER = power_ocsfull, method = 'OptCS-Full-MSel', n = nfull, alpha = alpha))
  
  
  
  return(data)
}
stopCluster(cl)

save(Result, file = 'Mammo-re.RData')


pp <- Result%>%
  group_by(method)%>%
  dplyr::summarize(t = mean(t), FDR = mean(FDP), power = mean(POWER), sdFDR = sd(FDP)/sqrt(nr), sdpower = sd(POWER)/sqrt(nr))
pp
