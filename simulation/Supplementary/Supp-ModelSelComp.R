library(MASS)
library(foreach)
library(randomForest)
library(doParallel)
library(e1071)
library(isotree)

source("functions.R")




m <- 1000
alpha <- 0.1
d <- 50


sig <- 1
nfull <- 300
piarray <- c(0.95, 0.93, 0.91, 0.89, 0.87, 0.85)


algoarray <- data.frame(matrix(NA, nrow = 6, ncol = 2))
names(algoarray) <- c('algotype', 'param')
algoarray$algotype <- rep(c('RF', 'SVM'), each = 3)
algoarray$param <- c(100, 300, 500, 0.1, 0.2, 0.4)


nr <- 500
Resultall <- data.frame()
for (pivalue in piarray) {

  
  cl <- makeCluster(10)
  registerDoParallel(cl)
  Result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "isotree", "e1071", "randomForest"), .errorhandling = "remove")%dopar% {
    
    data <- data.frame()
    
    pi0 <- pivalue
    pi0l <- pivalue
    
    n <- nfull*pi0l - 1
    n1 <- nfull - n
    
    
    X0 <- mvrnorm(n, rep(0, d), diag(d))
    X1 <- mvrnorm(n1, c(rep(sig*sqrt(log(d)), 5), rep(0, d-5)), diag(d))
    X_test <- rbind(mvrnorm(pi0*m, rep(0, d), diag(d)), mvrnorm(m-pi0*m, c(rep(sig*sqrt(log(d)), 5), rep(0, d-5)), diag(d)))
    outlier <- (pi0*m+1):m
    
    
    X0_train <- X0[1:(n/2),]
    X0_cal <- X0[((n/2)+1):n,]
    X1_train <- X1[1:(n1/2),]
    X1_cal <- X1[((n1/2)+1):n1,]
    
    m_pure <- ceiling(m/2)
    
    
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
    t_ecot_pure <- 0
    ecot_pure_model_list <- list()
    pval_pure_mat <- matrix(NA, nrow = m, ncol = nrow(algoarray))
    Vncal_pure_mat <- matrix(NA, nrow = n, ncol = nrow(algoarray))
    Vntest_pure_mat <- matrix(NA, nrow = m, ncol = nrow(algoarray))
    for (i in 1:nrow(algoarray)) {
      if (algoarray$algotype[i]=='RF'){
        t_before <- proc.time()
        ecot_temp_model <- randomForest(y~., data = data.frame(x = rbind(X0, X_test, X1), y = factor(c(rep(0, n+m), rep(1, n1)))), ntree = algoarray$param[i])
        t_after <- proc.time()
        t_ecot <- t_ecot + (t_after - t_before)['elapsed']
        
        
        t_before <- proc.time()
        V_idf_temp <- predict(ecot_temp_model, data.frame(x = rbind(X0, X_test)), type = 'prob')[, 2]
        if(sum(V_idf_temp==sort(V_idf_temp)[n+m_pure])==1){
          X0_pure_temp <- rbind(X0, X_test)[which(V_idf_temp<=sort(V_idf_temp)[n+m_pure]), ]
        }else{
          X0_pure_temp <- rbind(X0, X_test)[c(which(V_idf_temp<sort(V_idf_temp)[n+m_pure]), sample(which(V_idf_temp==sort(V_idf_temp)[n+m_pure]), n+m_pure-length(which(V_idf_temp<sort(V_idf_temp)[n+m_pure])))), ]
        }
        ecot_temp_model_pure <- randomForest(y~., data = data.frame(x = rbind(X0_pure_temp, X1), y = factor(c(rep(0, n+m_pure), rep(1, n1)))), ntree = algoarray$param[i])
        ecot_pure_model_list[[i]] <- list(model = ecot_temp_model_pure, type = algoarray$algotype[i], param = algoarray$param[i])
        Vncal_pure_mat[, i] <- predict(ecot_temp_model_pure, data.frame(x = X0), type = 'prob')[, 2]
        Vntest_pure_mat[ ,i] <- predict(ecot_temp_model_pure, data.frame(x = X_test), type = 'prob')[, 2]
        pval_pure_mat[, i] <- sapply(Vntest_pure_mat[ ,i], function(x){(sum(x<=Vncal_pure_mat[, i])+1)/(length(Vncal_pure_mat[, i])+1)})
        t_after <- proc.time()
        t_ecot_pure <- t_ecot_pure + (t_after - t_before)['elapsed']
      }
      if (algoarray$algotype[i]=='SVM'){
        t_before <- proc.time()
        ecot_temp_model <- svm(y~., data = data.frame(x = rbind(X0, X_test, X1), y = factor(c(rep(0, n+m), rep(1, n1)))), probability = T, nu = algoarray$param[i])
        t_after <- proc.time()
        t_ecot_pure <- t_ecot_pure + (t_after - t_before)['elapsed']
        
        
        t_before <- proc.time()
        V_idf_temp <- attr(predict(ecot_temp_model, data.frame(x = rbind(X0, X_test)), probability = T), 'probabilities')[, 2]
        if(sum(V_idf_temp==sort(V_idf_temp)[n+m_pure])==1){
          X0_pure_temp <- rbind(X0, X_test)[which(V_idf_temp<=sort(V_idf_temp)[n+m_pure]), ]
        }else{
          X0_pure_temp <- rbind(X0, X_test)[c(which(V_idf_temp<sort(V_idf_temp)[n+m_pure]), sample(which(V_idf_temp==sort(V_idf_temp)[n+m_pure]), n+m_pure-length(which(V_idf_temp<sort(V_idf_temp)[n+m_pure])))), ]
        }
        ecot_temp_model_pure <- svm(y~., data = data.frame(x = rbind(X0_pure_temp, X1), y = factor(c(rep(0, n+m_pure), rep(1, n1)))), probability = T, nu = algoarray$param[i])
        ecot_pure_model_list[[i]] <- list(model = ecot_temp_model_pure, type = algoarray$algotype[i], param = algoarray$param[i])
        Vncal_pure_mat[, i] <- attr(predict(ecot_temp_model_pure, data.frame(x = X0), probability = T), 'probabilities')[, 2]
        Vntest_pure_mat[, i] <- attr(predict(ecot_temp_model_pure, data.frame(x = X_test), probability = T), 'probabilities')[, 2]
        pval_pure_mat[, i] <- sapply(Vntest_pure_mat[ ,i], function(x){(sum(x<=Vncal_pure_mat[, i])+1)/(length(Vncal_pure_mat[, i])+1)})
        t_after <- proc.time()
        t_ecot_pure <- t_ecot_pure + (t_after - t_before)['elapsed']
      }
    }
    t_ecot_pure <- t_ecot_pure + t_ecot
    
    
    model_RF_B <- randomForest(y~., data = data.frame(x = rbind(X0_train, X1_train), y = factor(c(rep(0, n/2), rep(1, n1/2)))), ntree = 500)
    model_RF_new <- randomForest(y~., data = data.frame(x = rbind(X0, X_test, X1), y = factor(c(rep(0, n+m), rep(1, n1)))), ntree = 500)
    
    
    V_idf <- predict(model_RF_new, data.frame(x = rbind(X0, X_test)), type = 'prob')[, 2]
    if(sum(V_idf==sort(V_idf)[n+m_pure])==1){
      X0_pure <- rbind(X0, X_test)[which(V_idf<=sort(V_idf)[n+m_pure]), ]
    }else{
      X0_pure <- rbind(X0, X_test)[c(which(V_idf<sort(V_idf)[n+m_pure]), sample(which(V_idf==sort(V_idf)[n+m_pure]), n+m_pure-length(which(V_idf<sort(V_idf)[n+m_pure])))), ]
    }
    model_RF_new_pure <- randomForest(y~., data = data.frame(x = rbind(X0_pure, X1), y = factor(c(rep(0, n+m_pure), rep(1, n1)))), ntree = 500)
    
    
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
    data <- rbind(data, data.frame(t = t_ocs, FDP = FDP_B_sel, POWER = power_B_sel, method = 'OptCS-MSel', n = nfull, sig = sig, alpha = alpha, pivalue = pivalue))
    
    
    ###---ECOT-pure method selection---###
    rejnum_sel_pure <- rep(NA, m)
    pval_sel_pure <- rep(NA, m)
    ind_vec_pure <- rep(NA, m)
    t_before <- proc.time()
    for (j in 1:m){
      rejnum_each_pure <- rep(NA, nrow(algoarray))
      rejnum_each_mod_pure <- rep(NA, nrow(algoarray))
      
      for (k in 1:nrow(algoarray)) {
        Vn_pure_cal_sel <- c(Vncal_pure_mat[, k], Vntest_pure_mat[j, k])
        rem_bi_pure <- which.max(Vn_pure_cal_sel)
        Vn_pure_cal_sel <- Vn_pure_cal_sel[-rem_bi_pure]
        Vn_pure_test_sel <- Vntest_pure_mat[-j, k]
        
        pval_pure_new_sel <- sapply(Vn_pure_test_sel, function(x){(sum(x<=Vn_pure_cal_sel)+1)/(length(Vn_pure_cal_sel)+1)})
        if(j==1){
          pval_pure_new_sel <- c(0, pval_pure_new_sel)
        }else if(j==m){
          pval_pure_new_sel <- c(pval_pure_new_sel, 0)
        }else {
          pval_pure_new_sel <- c(pval_pure_new_sel[1:(j-1)], 0, pval_pure_new_sel[j:(m-1)])
        }
        pval_pure_new_sel_mod <- sapply(pval_pure_new_sel, function(x){x-1/(length(Vn_pure_cal_sel)+1)})
        pval_pure_new_sel <- pval_pure_new_sel*pi0l
        pval_pure_new_sel_mod <- pval_pure_new_sel_mod*pi0l
        pval_pure_new_sel_mod[j] <- 0
        rejnum_each_pure[k] <- length(BH(pval_pure_new_sel, alpha))
        rejnum_each_mod_pure[k] <- length(BH(pval_pure_new_sel_mod, alpha))
      }
      
      if(sum(rejnum_each_pure==1)<nrow(algoarray)){
        if(sum(rejnum_each_pure==max(rejnum_each_pure))==1){
          ind_pure <- which(rejnum_each_pure==max(rejnum_each_pure))
          rejnum_sel_pure[j] <- max(rejnum_each_pure)
        }else {
          if(sum(rejnum_each_mod_pure==max(rejnum_each_mod_pure))==1){
            ind_pure <- which(rejnum_each_mod_pure==max(rejnum_each_mod_pure))
          }else {
            ind_pure <- sample(which(rejnum_each_mod_pure==max(rejnum_each_mod_pure)), 1)
          }
          rejnum_sel_pure[j] <- max(rejnum_each_mod_pure)
        }
      }else {
        if(sum(rejnum_each_mod_pure==max(rejnum_each_mod_pure))==1){
          ind_pure <- which(rejnum_each_mod_pure==max(rejnum_each_mod_pure))
        }else {
          ind_pure <- sample(which(rejnum_each_mod_pure==max(rejnum_each_mod_pure)), 1)
        }
        rejnum_sel_pure[j] <- max(rejnum_each_mod_pure)
      }
      
      
      ind_vec_pure[j] <- ind_pure
      pval_sel_pure[j] <- pval_pure_mat[j, ind_pure]*pi0l
    }
    rej_sel_pure_init <- which(pval_sel_pure<=alpha*rejnum_sel_pure/m)
    if(length(rej_sel_pure_init)>=max(rejnum_sel_pure)){
      rej_sel_pure <- rej_sel_pure_init
    }else {
      rej_sel_pure <- rej_sel_pure_init[BH(runif(length(rej_sel_pure_init))*rejnum_sel_pure[rej_sel_pure_init]/length(rej_sel_pure_init), 1)]
    }
    rej_sel_pure <- BH(pval_sel_pure, alpha)
    t_after <- proc.time()
    t_ecot_pure <- t_ecot_pure + (t_after - t_before)["elapsed"]
    power_sel_pure <- sum(rej_sel_pure%in%outlier)/length(outlier)
    FDP_sel_pure <- sum(!rej_sel_pure%in%outlier)/max(length(rej_sel_pure), 1)
    data <- rbind(data, data.frame(t = t_ecot_pure, FDP = FDP_sel_pure, POWER = power_sel_pure, method = 'ECOT-as', n = nfull, sig = sig, alpha = alpha, pivalue = pivalue))
    
    
    ###---OCS-full---###
    pval_ocsfull <- rep(NA, m)
    ind_full_vec <- rep(NA, m)
    Vfullcal_mat <- matrix(NA, nrow = nfull, ncol = nrow(algoarray))
    Vfulltest_mat <- matrix(NA, nrow = m, ncol = nrow(algoarray))
    t_ocs_full_model <- 0
    t_before <- proc.time()
    t_before_fullmodel <- proc.time()
    for (j in 1:(nfull+m)) {
      for (i in 1:nrow(algoarray)) {
        if (algoarray$algotype[i]=='RF'){
          ocsfull_temp_model <- randomForest(y~., data = data.frame(x = rbind(X0, X1, X_test)[-j, ], y = factor(c(rep(0, n), rep(1, n1), rep(0, m))[-j])), ntree = algoarray$param[i])
          if (j<=nfull) {
            Vfullcal_mat[j, i] <- predict(ocsfull_temp_model, data.frame(x = t(rbind(X0, X1, X_test)[j, ])), type = 'prob')[2] - c(rep(0, n), rep(1, n1), rep(0, m))[j]*1000
          } else {
            Vfulltest_mat[j-nfull, i] <- predict(ocsfull_temp_model, data.frame(x = t(rbind(X0, X1, X_test)[j, ])), type = 'prob')[2]
          }
        }
        if (algoarray$algotype[i]=='SVM'){
          ocsfull_temp_model <- svm(y~., data = data.frame(x = rbind(X0, X1, X_test)[-j, ], y = factor(c(rep(0, n), rep(1, n1), rep(0, m))[-j])), probability = T, nu = algoarray$param[i])
          if (j<=nfull) {
            Vfullcal_mat[j, i] <- attr(predict(ocsfull_temp_model, data.frame(x = t(rbind(X0, X1, X_test)[j, ])), probability = T), 'probabilities')[2] - c(rep(0, n), rep(1, n1), rep(0, m))[j]*1000
          } else {
            Vfulltest_mat[j-nfull, i] <- attr(predict(ocsfull_temp_model, data.frame(x = t(rbind(X0, X1, X_test)[j, ])), probability = T), 'probabilities')[2]
          }
        }
      }
    }
    t_after_fullmodel <- proc.time()
    t_ocs_full_model <- (t_after_fullmodel - t_before_fullmodel)['elapsed']
    
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
    data <- rbind(data, data.frame(t = t_ocs_full, FDP = FDP_ocsfull, POWER = power_ocsfull, method = 'OptCS-Full-MSel', n = nfull, sig = sig, alpha = alpha, pivalue = pivalue))
    
    return(data)
  }
  stopCluster(cl)
  
  Resultall <- rbind(Resultall, Result)
}

save(Resultall, file = 'ModelSelCompare.RData')


