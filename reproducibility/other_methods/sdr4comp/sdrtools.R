library(textir)
source("./auxtools/auxtools4sdr.R")
source("./auxtools/aTests.R")

scene4plot3d <- function(label){
  axe2dx <- list(
    title = paste0(label,"-1"),
    zeroline = FALSE,
    showline = TRUE,
    ticks = FALSE,
    showticklabels = FALSE,
    showgrid = TRUE,
    visible = TRUE
  )
  axe2dy <- list(
    title = paste0(label,"-2"),
    zeroline = FALSE,
    showline = TRUE,
    ticks = FALSE,
    showticklabels = FALSE,
    showgrid = TRUE,
    visible = TRUE
  )
  axe2dz <- list(
    title = paste0(label,"-3"),
    zeroline = FALSE,
    showline = TRUE,
    ticks = FALSE,
    showticklabels = FALSE,
    showgrid = TRUE,
    visible = TRUE
  )
  myscene = list(
    xaxis = axe2dx,
    yaxis = axe2dy,
    zaxis = axe2dz,
    camera = list(eye = list(x = -1.25, y = 1.25, z = 1.25))
  )
  return(myscene)
}

checkData <- function(compositions,totals,labels){
  X = as.matrix(compositions)
  reads = as.matrix(totals)
  q = ncol(X)
  counts = matrix(NA,nrow(X),ncol(X))
  for (j in 1:q){
    counts[,j] = round(X[,j]*reads)
  }
  
  idx0 = which(apply(counts,2,sum)<=5)
  X.counts = counts[,apply(counts,2,sum)>5]
  idx = which(apply(X.counts,1,sum)<10)
  X.counts = X.counts[-idx,]
  Y = as.matrix(labels)
  Y = Y[-idx]
  X = X[-idx,]
  outputs = list(compositions=X,counts = X.counts, labels = Y)
  return(outputs)
}

sdr4multinomial.fit <- function(x,y,dim=ncol(get_fyZ(y)),lambda=0){
  # compute invariant initial estimate
  XX.counts = ceil(x/apply(x,1,sum)*10000)
  fy = get_fyZ(y)
  mn.fit = dmr(cl=NULL,covars=fy,counts=XX.counts)
  bb.dmr = t(as.matrix(coef(mn.fit)))

  if (lambda > 0){
    q = nrow(bb.dmr)
    bb.dmr = sweep(bb.dmr[1:(q-1),],2,bb.dmr[q,])
    TT = (matrix(-1,q-1,q-1) + q*diag(q-1))/q
    beta = TT%*%bb.dmr
    beta0 = rbind(beta,-1*as.numeric(apply(beta,2,sum)))
  
    # compute svd
    bMN = beta[,-1]
    aux = svd(bMN)
    bb = aux$u%*%diag(aux$d)
    beta.svd = TT%*%bb
    beta.svd = rbind(beta.svd,-1*as.numeric(apply(beta.svd,2,sum)))

  # variable selection  
    bb0 = beta0
    bb1 = myFullGrpLasso(XX.counts,cbind(rep(1,length(y)),fy),bb0,lambdaInit=0.0,lambda=lambda,ccInit=0.0,cc=1.0,r=0.5,model="mult",doweight=FALSE)
    bhat = bb1$b[,-1]

    # find the selected variables  
    idxsel = which(apply(bhat,1,Norm)>0.0000001)
    bb.inv = sweep(bhat[1:(q-1),],2,bhat[q,])
    beta = TT%*%bb.inv
    beta = rbind(beta,-1*as.numeric(apply(beta,2,sum)))
    aux = matrix(0.0,ncol=ncol(bhat),nrow=nrow(bhat))
    aux[idxsel,] = scale(beta[idxsel,],scale=FALSE)
    bhat = aux
  }else{bhat = bb.dmr}
    if (ncol(fy) > 2) {bhat[,1:3] = bhat[,c(1,3,2)]; bhat = bhat[,1:dim];}
    newproj = XX.counts%*%bhat
    proj = cbind(newproj,apply(x,1,sum))
    results = list(bhat = bhat, proj = proj)
    return(results)
}

sdr4multinomial.project <- function(fit,x){
  XX.counts = ceil(x/apply(x,1,sum)*10000)
  proj = XX.counts%*%fit$bhat
  proj = cbind(proj,apply(x,1,sum))
  return(proj)
}

sdr4normal.fit <- function(x,y,dim=ncol(get_fyZ(y)),lambda=0){
  ##---------------------- NORMAL-------------------------------
  totals = apply(x,2,sum)
  base = which.max(totals)
  ratios = Ini.Logratios(base,x)
  
  ratios.c = scale(ratios,scale=FALSE)
  fy = get_fyZ(y)
  m = lm(ratios.c~fy)
  b = as.matrix(coef(m))[-1,]
  if (is.numeric(b)){
    b = t(as.matrix(b))
  }

  if (is(b, "matrix")){
    b = t(b)
    if (dim < ncol(fy)){
    b = regularize(b,dim,lambda)
    # aux = svd(b)
    # newb = aux$v%*%diag(aux$d)
    # normas = apply(newb,1,Norm)
    # idx = which(normas < lambda)
    # newb[idx,] = 0 
    }
  }else{
    normas = abs(b) # Error in the source code; fixed newb -> b
    idx = which(normas < lambda)
    b[idx] = 0 
  } 
  aux = matrix(0,ncol=ncol(b),nrow=nrow(b)+1)
  aux[base,] = -1*as.numeric(apply(b,2,sum))
  aux[-base,] = b
  newb = aux
  newproj = log(x+.1)%*%newb
  resultados = list(bhat = newb, proj = newproj, base = base, b=b)
  return(resultados)
}

sdr4normal.project <- function(fit,xnew){
  ratios = Ini.Logratios(fit$base,xnew)
  newproj = ratios%*%fit$bhat[-fit$base,]
  return(newproj)
}

sdr4pgmR.fit <- function(x,y,d=ncol(get_fyZ(y)),lambda=0.0){
  XX.counts = ceil(x/apply(x,1,sum)*10000)
  fy = get_fyZ(y)
  fit0 = PGM.invreg(X=XX.counts,Z=fy,epsilon = 1e-1)
  Bsvd=svd(fit0$B[,-1], nu = ncol(fy), nv = ncol(fy))
  bhat = Bsvd$u%*%diag(Bsvd$d[1:ncol(fy)]); 
  if (d < ncol(fy)){
    bhat = regularize(bhat,d+1,lambda)
    bhat = swap(bhat,d,d+1); bhat = bhat[,1:d] 
  }
  proj = sqrt(XX.counts)%*%bhat
  proj = cbind(proj,apply(x,1,sum))
  return(list(bhat = bhat, proj = proj))
}


sdr4pgmR.project <- function(fit, x){
  XX.counts = ceil(x/apply(x,1,sum)*10000)
  proj = sqrt(XX.counts)%*%fit$bhat
  proj = cbind(proj,apply(x,1,sum))
  return(proj)
}

regularize <- function(A,d,lambda = 0){
  p=dim(A)[1]
  q=dim(A)[2]
  Asvd=svd(A, nu = d, nv = d)
  x=kronecker(Asvd$v,diag(rep(1,p)))
  y=vec(A)
  groups = vec(matrix(1:p,p,1)%*%matrix(rep(1,d),1,d))
  fit    <- grplasso(x,y=y,model = LinReg(),index = groups,
                     lambda = lambda,
                     coef.init = vec(Asvd$u%*%diag(Asvd$d[1:d])),
                     center = FALSE,standardize = FALSE)$coefficients
  dim(fit)=c(p,d)
  return(fit)
}


swap <- function(x,i,j){
  if (ncol(x)<2){
    out = x
  }else{
    out = x
    out[,c(i,j)]=x[,c(j,i)]
  }
  return(out)
}

testDim4sdr <- function(x,y,model,alpha=0.05){
  if (model=="normal"){ d = testDim(x,y,alpha)}
  else if(model=="multinomial"){d=testDimMN(x,y,alpha)}
  else if(model=="PGM"){d=testDimPGM(x,y,alpha)}
  else{print("specifief model not available");d=get_fyZ(y);}
  return(d)
}