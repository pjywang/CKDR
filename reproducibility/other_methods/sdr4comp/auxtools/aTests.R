library(matrixcalc)
library(momentchi2)
library(pracma)
aVar <- function(x,y,method){
 if(method=="normal"){
   totals = apply(x,2,sum)
   base = which.max(totals)
   ratios = Ini.Logratios(base,x)
   ratios.c = scale(ratios,scale=FALSE)
   fy = get_fyZ(y)
   r = ncol(fy)
   m = lm(ratios.c~fy)
   b = as.matrix(coef(m))
   idelta = solve(cov(m$residuals))
   B = t(b%*%idelta)
   Ct = cbind(rep(0.0,r),diag(r))
   S = pinv(cov(cbind(rep(1,nrow(fy)),fy)))
   V = kron(S,idelta) + kron(t(B)%*%cov(m$residuals)%*%B,idelta) + kron(t(B),B)%*%commutation.matrix(ncol(ratios),ncol(fy)+1)
   out = kron(Ct,diag(ncol(ratios)))%*%V%*%kron(t(Ct),diag(ncol(ratios)))
 }else if (method=="multinomial"){
   fy = get_fyZ(y)
   r = ncol(fy)
   k = ncol(x)
   mn.fit = dmr(cl=NULL,covars=fy,counts=x)
   bb.dmr = t(as.matrix(coef(mn.fit)))  
   q = nrow(bb.dmr)
   bb.dmr = sweep(bb.dmr[1:(q-1),],2,bb.dmr[q,])
   Ct = cbind(rep(0.0,r),diag(r))
   Fy = cbind(rep(1,nrow(fy)),fy)
   ETA = Fy%*%t(bb.dmr)
   H = getH4mn(ETA,y)
   iV = getV4mn(Fy,H,y)
   out = kron(Ct,diag(k-1))%*%pinv(iV)%*%kron(t(Ct),diag(k-1))
 }else if (method =="PGM"){
   fy = get_fyZ(y)
   r = ncol(fy)
   k = ncol(x)
   mn.fit = dmr(cl=NULL,covars=fy,counts=x)
   bb.dmr = t(as.matrix(coef(mn.fit)))  
   q = nrow(bb.dmr)
   bb.dmr = sweep(bb.dmr[1:(q-1),],2,bb.dmr[q,])
   Ct = cbind(rep(0.0,r),diag(r))
   Fy = cbind(rep(1,nrow(fy)),fy)
   ETA = Fy%*%t(bb.dmr)
   H = getH4mn(ETA,y)
   iV = getV4mn(Fy,H,y)
   out = kron(Ct,diag(k-1))%*%V%*%kron(t(Ct),diag(k-1))
 }else{
    return(1)
 }
  return(out)
}

testDim <- function(x,y,alpha=0.05){
  r = ncol(get_fyZ(y))
  fit = sdr4normal.fit(x,y,r)
  b = fit$bhat[-fit$base,]
  b.svd = svd(b)
  Gamma = aVar(x,y,"normal")
  V0 = b.svd$u #[,-c(1)]
  L0 = b.svd$v #[,-c(1)]
  Q = kron(t(L0),t(V0))%*%Gamma%*%kron(L0,V0)
    for (m in 1:min(r,ncol(x)-1)){
    # V0 = b.svd$u[,-c(1:m)]
    # L0 = b.svd$v[,-c(1:m)]
    # Q = kron(t(L0),t(V0))%*%Gamma%*%kron(L0,V0)
    statistic = sum(b.svd$d[-c(1:m)]^2)
    pval = hbe(eigen(Q)$values[1:m],statistic)
    if (pval < alpha){
      dimension = m-1
      break
    }
  }
  return(dimension)
}


testDimMN <- function(x,y,alpha=0.05){
  fy = get_fyZ(y)
  r = ncol(fy)
  k = ncol(x)
  mn.fit = dmr(cl=NULL,covars=fy,counts=x)
  bb.dmr = t(as.matrix(coef(mn.fit)))  
  q = nrow(bb.dmr)
  bb.dmr = sweep(bb.dmr[1:(q-1),],2,bb.dmr[q,])[,-1]
  b.svd = svd(bb.dmr)
  Gamma = aVar(x,y,"multinomial")
  V0 = b.svd$u #[,-c(1)]
  L0 = b.svd$v #[,-c(1)]
  Q = kron(t(L0),t(V0))%*%Gamma%*%kron(L0,V0)
  for (m in 1:min(r,ncol(x)-1)){
    statistic = sum(b.svd$d[-c(1:m)]^2)
    pval = hbe(eigen(Q)$values[1:m],statistic)
    if (pval < alpha){
      dimension = m+1
      break
    }
  }
  return(dimension)
}


testDimPGM <- function(x,y,alpha=0.05){
  fy = get_fyZ(y)
  r = ncol(fy)
  k = ncol(x)
  mn.fit = dmr(cl=NULL,covars=fy,counts=x)
  bb.dmr = t(as.matrix(coef(mn.fit)))  
  q = nrow(bb.dmr)
  bb.dmr = sweep(bb.dmr[1:(q-1),],2,bb.dmr[q,])[,-1]
  b.svd = svd(bb.dmr)
  Gamma = aVar(x,y,"multinomial")
  V0 = b.svd$u #[,-c(1)]
  L0 = b.svd$v #[,-c(1)]
  Q = kron(t(L0),t(V0))%*%Gamma%*%kron(L0,V0)
  for (m in 2:min(r,ncol(x)-1)){
    statistic = sum(b.svd$d[-c(1:m-1)]^2)
    pval = hbe(eigen(Q)$values[1:m-1],statistic)
    if (pval < alpha){
      dimension = m+1
      break
    }
  }
  return(dimension)
}


getH4mn <- function(eta,y){
  yvals = unique(y)
  h = length(yvals)
  Hmtx = array(NA,dim=c(h,length(eta[1,]),length(eta[1,])))
  for (j in 1:h){
    idx = which(y==yvals[j])
    etay = eta[idx[1],]
    suma = sum(exp(etay))
    Hy = matrix(0,length(etay),length(etay))
    for (k in 1:length(etay)){
      Hy[k,k] = exp(etay[k])/(1+suma) - exp(2*etay[k])/(1+suma)^2
      if (k < length(etay)){
        for (l in (k+1):length(etay)){
          Hy[l,k] = - exp(etay[l]+etay[k])/(1+suma)^2
          Hy[k,l] = Hy[l,k]
        }
      }
    }
    Hmtx[j,,]=Hy
  }
  return(Hmtx)
}


getV4mn <- function(Fy,H,y){
  yvals = unique(y)
  h = length(yvals)
  suma = matrix(0.0,ncol(Fy)*ncol(H),ncol(Fy)*ncol(H))
  for (j in 1:h){
    idx = which(y==yvals[j])
    Fyj = Fy[idx[1],]
    dim(Fyj) = c(1,length(Fyj))
    aux = kron(t(Fyj)%*%Fyj,H[j,,])
    suma = suma + length(idx)*aux
  }
  suma = suma/length(y)
  return(suma)
}