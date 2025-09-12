BH<-function(pval,alpha){
  m=length(pval)
  rej <- sort(pval)<((1:m)/m)*alpha
  rejnum <- max(which(rej==T))
  reject <- which(pval<=sort(pval)[rejnum])
  return(reject)
}

Result_Report<-function(reject,nonnull){
  FDP <- (length(reject) - sum(reject%in%nonnull))/max(c(length(reject), 1))
  power <- sum(reject%in%nonnull)/length(nonnull)
  return(data.frame(FDP = FDP, power = power))
}


NullIndex <- function(y, Value){
  if(Value$type=="==A"){
    index <- which(y==Value$v)
  }else if(Value$type=="<=A"){
    index <- which(y<=Value$v)
  }else if(Value$type==">=B"){
    index <- which(y>=Value$v)
  }else if(Value$type=="<=A|>=B"){
    index <- which(y<=Value$v[1]|y>=Value$v[2])
  }else if(Value$type==">=A&<=B"){
    index <- which(y>=Value$v[1]&y<=Value$v[2])
  }
  return(index)
}

Conditional_Calibration<-function(pval,R,alpha){
  m<-length(pval)
  rej_cc_pre <- pval<=(R*alpha/m)
  rejnum_cc_pre <- sum(rej_cc_pre)
  if (rejnum_cc_pre<=min(R)) {
    rej_cc <- rej_cc_pre
  } else {
    r <- 1
    eps <- runif(m)
    for (rr in 1:rejnum_cc_pre) {
      if (sum(eps[rej_cc_pre]*R[rej_cc_pre]<=rr)>=rr) {
        r <- rr
      }
    }
    rej_cc <- (pval<=(R*alpha/m))&(eps*R<r)
  }
  reject_cc <- which(rej_cc==T)
  return(reject_cc)
}