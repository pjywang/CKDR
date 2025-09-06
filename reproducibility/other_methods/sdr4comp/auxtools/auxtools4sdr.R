#####################################################################################
###           SDR FOR COMPOSITIONAL DATA				    ###
###           Date: 10_2019                                                       ###
###           AUX TOOLS FOR SDR FOR COMPOSITIONAL DATA               ###
#####################################################################################

require(dirmult)
require(MASS)
require(glmnet)
require(pracma)
library(grplasso)


#############################################################
Loglik <- function(Y, X, b,  model) {
	g <- exp(X %*% t(b))	# n * q
	gs <- rowSums(g)
	ys <- rowSums(Y)
	if (model == "dirmult") {
		res <- 	sum(lgamma(gs) - lgamma(ys+gs) +
						rowSums(lgamma(Y+g) - lgamma(g)))
	}
  if (model == "mult") {
		res <- 	sum(rowSums(Y * log(g)) - ys * log(gs))
	}
  if (model == "dir") {
		res <- 	sum(lgamma(gs) + rowSums((g-1) * log(Y) - lgamma(g)))
	}

  res / nrow(X)
}

ObjFunc <- function(bi, i, b0, Y, X, c1=0, c2=0, model){
	# Objective function: -Loglik + group lasso + lasso for ith group
  #
  # Fills a column
	b0[, i] <- bi
	-Loglik(Y, X, b0, model) + c1*sqrt(sum(bi^2)) + c2*sum(abs(bi))
}


ObjFunc2 <- function(bi, i, b0, Y, X, c1=0, c2=0, model){
  # Objective function: -Loglik + group lasso + lasso for ith group
  #
  # Fills a row
  b0[i,] <- bi
  -Loglik(Y, X, b0, model) + c1*sqrt(sum(bi[-1]^2)) + c2*sum(abs(bi[-1]))
}


S <- function(Y, X, b, model) {
	S <- 0
	g <- exp(X %*% t(b))	# n * q
	gs <- rowSums(g)
	ys <- rowSums(Y)
	
	if (model == "dirmult") {
		S <-  t((digamma(gs) - digamma(ys+gs) +
					digamma(Y+g) - digamma(g)) * g) %*% X
	}
  
	if (model == "mult") {
		S <- t((Y / g - ys / gs) * g) %*% X
	}
  
	if (model == "dir") {
		S <-  t((digamma(gs)  - digamma(g) + log(Y)) * g) %*% X
	}
	S / nrow(X)
}

H <- function(Y, X, b, model){
	H <- 0
	g <- exp(X %*% t(b))	# n * q
	gs <- rowSums(g)
	ys <- rowSums(Y)
	if (model == "dirmult") {
			H <- t((trigamma(gs) - trigamma(ys+gs) +
						trigamma(Y+g) - trigamma(g)) * g^2 +
					(digamma(gs) - digamma(ys+gs) +
						digamma(Y+g) - digamma(g)) * g) %*% X^2
	}
  if (model == "mult") {
			H <- t((-Y / g^2 + ys / gs^2) * g^2  +
					(Y / g - ys / gs) * g) %*% X^2
	}
	if (model == "dir") {
			H <- t((trigamma(gs) - trigamma(g)) * g^2  +
					(digamma(gs)  - digamma(g) + log(Y)) * g) %*% X^2
	}
	H / nrow(X)
}
#############################################################
myDirmultGrp <- function(Y, X, b0, lamda=0, cc=0, model="dirmult",
		alpha0   = 1,
		delta    = 0.5,
		sigma    = 0.1,
		cstar    = 0.001,
		tol     = 1e-4,
		iter.lim = 500) {
	# Minimize objective function: -Loglik +  group lasso + lasso

	b0 <- as.matrix(b0)		
	b00 <- b0
	n <- nrow(X)
	p <- ncol(b0)
	q <- nrow(b0)
	c1 = 0
	c2 = 0

	# check inputs------------------------------
	if (ncol(Y) != q || ncol(X) != p) stop("Dimension does not match!\n")
	if (sum(X[, 1] != 1) != 0) {
		warning("No intercepts in X! Intercepts added\n")
		X <- cbind(rep(1, nrow(X)), X)
	}
	if (nrow(Y) != nrow(X)) stop("Y and X have different sample sizes!\n")	
	if (model == "dir") {	
		if (max(Y) > 1) {
			Y <- (Y + 0.5) / rowSums(Y)
		} else {
			if (min(Y) == 0) {
				Y[Y==0] <- min(Y[Y!=0])/10
			}
		}
	}
	
	
	#-------------- MAIN LOOP--------------------------------
	b1 <- b0
	iter <- 0
	
	repeat {
		b0 <- b1
		h <- H(Y, X, b0, model)
		s <- S(Y, X, b0, model)
		
		h1 <- -max(-h[, 1], cstar) 
		d1 <- -s[, 1] / h1         
		if (sum(d1 != 0) != 0){
			Delta <- - sum(d1 * s[, 1])
			alpha <- alpha0
			alpha.d1 <- alpha * d1
			while (ObjFunc(b1[, 1] + alpha.d1, 1, b1, Y, X, model=model) - 
					ObjFunc(b1[, 1], 1, b1, Y, X, model=model) > alpha*sigma*Delta) {
				alpha <- alpha * delta
				alpha.d1 <- alpha * d1
				if (max(abs(alpha.d1)) < 0.1*tol){alpha.d1 <- 0; break}
			}
			b1[, 1] <- b1[, 1] + alpha.d1
		}#-------------------------------------------------------end for the intercepts
		di <- numeric(q)
		if (p != 1){
			for (i in 2:p) {
				hi <- -max(-h[, i], cstar)
				diff.v <- s[, i]- hi * b1[, i] # see at the bottom of page 8.
				ind <- (abs(diff.v) <= c2)
				di[ind] <-  -b1[ind, i]	
				if (sum(ind) != q) {
					ind <- !ind				
					si.temp <- s[ind, i] - c2*(sign(s[, i] - hi*b1[, i])[ind])				
					diff.v <- si.temp - hi*b1[ind, i] #	dimension changed
					diff.n <- sqrt(sum(diff.v^2))
					if (diff.n <= c1){				
						di[ind]  <- -b1[ind, i]			
					} else {	
						di[ind] <- -(si.temp - c1*diff.v/diff.n) / hi
					}
				}

				if (sum(di != 0) != 0){
					Delta <- - sum(di * s[, i]) + 
							c1 * (sqrt(sum((b1[, i] + di)^2)) - sqrt(sum(b1[, i]^2))) +
							c2 * sum(abs(b1[, i] + di) - abs(b1[, i]))
					alpha <- alpha0
					alpha.di <- alpha * di
					while (ObjFunc(b1[, i] + alpha.di, i, b1, Y, X, c1, c2, model) - 
							ObjFunc(b1[, i], i, b1, Y, X, c1, c2, model) > alpha*sigma*Delta) {
						alpha <- alpha * delta
						alpha.di<- alpha * di
						if (max(abs(alpha.di)) < 0.1*tol){ alpha.di <- 0; break }
					}
					b1[, i] <- b1[, i] + alpha.di
				}
			}
		}
		
		iter <- iter + 1
		if (iter > iter.lim) {
		  break
		}
		if (max(abs(b1-b0)) <= tol) break
	}

	lik1 <- n * Loglik(Y, X, b1, model)
	
	
	
	if (lamda > 0){
	c1 <- sqrt(p-1) * lamda * cc # so c1, c2 are on the same level
  c2 = lamda*(1-cc)
	iter2 <- 0
	
	repeat {
	  b0 <- b1
	  h <- H(Y, X, b0, model)
	  s <- S(Y, X, b0, model)
	  
	   h1 <- -max(-h[, 1], cstar)
	   d1 <- -s[, 1] / h1
	   if (sum(d1 != 0) != 0){
	     Delta <- - sum(d1 * s[, 1])
	     alpha <- alpha0
	     alpha.d1 <- alpha * d1
	     
	     while (ObjFunc(b1[, 1] + alpha.d1, 1, b1, Y, X, model=model) - 
	            ObjFunc(b1[, 1], 1, b1, Y, X, model=model) > alpha*sigma*Delta) {
	       alpha <- alpha * delta
	       alpha.d1 <- alpha * d1
	       if (max(abs(alpha.d1)) < 0.1*tol){alpha.d1 <- 0; break}
	     }
	     b1[, 1] <- b1[, 1] + alpha.d1
	   }
	  
	  di <- matrix(0,p,1)
	    for (i in 1:q) {
	      hi <- -max(-h[i,-1], cstar)
	      diff.v <- s[i,-1]- hi * b1[i,-1]
	      ind <- (abs(diff.v) <= c2)
	      ind = c(FALSE,ind)
	      di[ind] <-  -b1[i,ind] ; di[1]=0	
	      if (sum(ind) != (p-1)) {
	        ind <- !ind				
	        ind[1] = FALSE
	        si.temp <- s[i,ind] - c2*(sign(s[i,] - hi*b1[i,])[ind])				
	        diff.v <- si.temp - hi*b1[i, ind] #	dimension changed
	        diff.n <- sqrt(sum(diff.v^2))
	        if (diff.n <= c1){				
	          di[ind]  <- -b1[i, ind]; 
	        } else {	
	          di[ind] <- -(si.temp - c1*diff.v/diff.n) / hi
	        }
	        di[1]=0
	      }
	      
	      if (sum(di != 0) != 0){
	        Delta <- - sum(di * s[i,]) + 
	          c1 * (sqrt(sum((b1[i,-1] + di[-1])^2)) - sqrt(sum(b1[i,-1]^2))) +
	          c2 * sum(abs(b1[i,-1] + di[-1]) - abs(b1[i,-1]))
	        alpha <- alpha0
	        alpha.di <- alpha * di
	        
	        b2 = b1[i,] + alpha.di
	        resta = ObjFunc2(b2, i, b1, Y, X, c1, c2, model) - ObjFunc2(b1[i,], i, b1, Y, X, c1, c2, model)
	        
	        if (is.nan(resta)){print('_____________SALIENDO POR nan________________'); print(iter2); break}
	        
	        bound = alpha*sigma*Delta
	        while (resta > bound) {
	          alpha <- alpha * delta
	          alpha.di<- alpha * di
	          if (max(abs(alpha.di)) < 0.1*tol){ 
	            alpha.di <- 0; 
	            break 
	          }
	          b2 = b1[i,] + alpha.di
	          resta = ObjFunc2(b2, i, b1, Y, X, c1, c2, model) - ObjFunc2(b1[i,], i, b1, Y, X, c1, c2, model)
	          if (is.nan(resta)){print('_____________SALIENDO POR nan________________'); print(iter2); break}
	        }
	        
	        b1[i,] <- b1[i,] + alpha.di
	      }
	    }

	  iter2 <- iter2 + 1
	  if (iter2 > 100) {
	    break
	  }
	  if (max(abs(b1-b0)) <= tol) break
	}
	}
	lik <- n * Loglik(Y, X, b1, model)
	lik0 <- n * Loglik(Y, X, b00, model)

	
	return(list(b=b1, lamda=lamda, cc=cc, iter=iter, loglik=lik, loglik1 = lik1))
}


####################################################################################

TuningMax <- function(Y, X, incpt, cc=0.00, model, st=NULL) {
	#
	# Returns:
	# 		lamda.max	
	if (is.null(st)) {
		if (model == "dirmult" | model == "dir") {
			st <- 4
		} else {
			st <- 50
		}
	}

	cat("Determine max lamda...\n")
	f <- function(lamda) {
		flag <- 0
		c1 <- lamda * cc * sqrt(q)
		c2 <- lamda * (1-cc)
		for (i in 2:p) {
			diff.v <- s[, i]
			ind <- (abs(diff.v) <= c2)
			if (sum(ind) == q) next
			ind <- !ind
			si.temp <- s[ind, i] - c2*(sign(s[, i])[ind])
			diff.v <- si.temp
			diff.n <- sqrt(sum(diff.v^2))
			if (diff.n > c1) {
				flag <- 1
				break
			}
		}
		flag
	}
	p <- ncol(X)
	q <- ncol(Y)
	b0 <- matrix(0, q, p)
	b0[, 1] <- incpt
	s <- S(Y, X, b0, model)
	lamda <- st
	flag <- 0
	while (flag == 0) {
		lamda <- lamda * 0.96
		flag <- f(lamda)
	}
	lamda / 0.96	
	print(lamda)
}

myDirmultGrpGrid <- function(counts, X, n.grid=30, lamda.max0=NULL, cc=0.20, nz=ncol(X),
		 model="dirmult", initscalar=10, ...) {
  print(initscalar)
	Y = counts
  res <- list()
	p <- ncol(X)
	q <- ncol(Y)
	
	print(lamda.max0)
	
	if (ncol(Y) != q || ncol(X) != p) stop("Dimension not match!\n")
	if (sum(X[, 1] != 1) != 0) {
		warning("No intercepts in X! Intercepts added\n")
		X <- cbind(rep(1, nrow(X)), X)
	}
	if (nrow(Y) != nrow(X)) stop("Y and X have different sample size!\n")
	if (sum(round(Y) != Y)) stop("Y should contain counts!\n")
	if (model == "dir") {	
		Y <- (Y + 0.5) / rowSums(Y)
	}	
	if (model == "dirmult") {
		cat("Sparse Dirichlet-Multinomial Regression\n")
	}
	if (model == "dir") {
		cat("Sparse Dirichlet Regression\n")
	}
	if (model == "mult") {
		cat("Sparse Multinomial Regression\n")
	}
	cat("Initial MLE of the coefficient matrix ...\n")
	b00 <- matrix(0, q, p,dimnames = list(dimnames(Y)[[2]],NULL))
	if (model == "dirmult" ) {
	  #=========================================
	  aux = dirmult(Y, initscalar = initscalar, trace=FALSE)
	  aux2 = log(aux$pi*aux$theta)
	  print(aux2)
    print(dim(b00))
	  b00[names(aux2),1] <- aux2
		#b00[, 1] <- log(dirmult(Y, trace=F)$gamma)
	} 
	if (model == "mult" | model == "dir") {
		f1 <- function(b) {
			b00[, 1] <- b
			-Loglik(Y, X, b00, model)
		}	
		b00[, 1] <- nlm(f1, rep(0, q))$estimate
	} 

	cat("Grid search:\n")
	ct <- 0
	for (j in 1:length(cc)) {
		b0 <- b00
		if(sum(is.null(lamda.max0)) != 0) {
			lamda.max <- TuningMax(Y, X, incpt=b00[, 1], cc=cc[j], model=model)
			lamda.range <- logspace(log10(lamda.max),-5,n=n.grid)
		} else if(length(lamda.max0) == 1) {
			lamda.range <- logspace(log10(lamda.max0),-5,n=n.grid)
		} else {
			lamda.range <- lamda.max0[j] * 0.96^(0:n.grid) 
		}
		

		for (i in 1:length(lamda.range)) {
			lamda <- lamda.range[i]
			print('El valor de lambda es:')
			print(lamda)
			# e <- try(obj <- myDirmultGrp(Y, X, b0, lamda, cc[j], model=model, ...))
#			e <- try(obj <- myFullGrpLasso(Y, X, b0, lamdaInit=0.0,lambda = lamda, ccInit=0.0, cc=cc[j], model=model, ...))
			obj <- myFullGrpLasso(Y, X, b0, lambdaInit=0.0,lambda = lamda, ccInit=0.0, cc=cc[j], model=model, ...)
#			if (class(e) == "try-error") stop("DirmultGrp error!")
			ct <- ct+1
			res[[ct]] <- obj
			b0 <- obj$b
			cat(".")
			# To restrict the number of paramter 
			if(sum(b0[, -1] != 0) >= nz) break
		}
		cat("\n")
	}
	cat("Finished!\n")
	res
}
#############################################################

DirmultSim <- function(n=100, nr=1000, s=2, rho=0.4,
		p=100, q=40, p.g=4, q.g=4, f=0.80, theta0=0.025){
	# Simulation strategy 
	theta <- theta0
	cat("Generating data ...\n")
	
	ct <- sample(nr:(2*nr), n, rep=T)
	
	Sigma <- matrix(1, p, p)
	Sigma <- rho^abs(row(Sigma)-col(Sigma))
	
	X <- cbind(rep(1, n), scale(mvrnorm(n=n, mu=rep(0, p), Sigma=Sigma)))
	Y <- matrix(0, n, q)
	b <- matrix(0, q, p)
	pi <- matrix(0, n, q)
	
	st <- 0
	if (q.g != 1) {
		coef <- seq(0.6, 0.9, len=q.g) * c(1, -1)	
	} else {
		coef <- (0.6 + 0.9) / 2
	}
	
	coef.g <- seq(1.0, 1.0, len=p.g)
	
	for (i in 1:p.g) {
		b[(st:(st+q.g-1))%%q+1, 3*i-2] <- coef.g[i] * coef[((i-1):(i+q.g-2))%%q.g+1]
		st <- st+1  
	}
	
	if (s==1) {
		gs <- (1-theta) / theta
		# base proportion max diff 100 fold
		icpt <- runif(q, 0.02, 2)
		# The theta for each sample is different, but let them close to supplied value
		icpt <- gs * icpt / sum(icpt) 
		icpt <- log(icpt)		
	} 
	if (s==2) {
		# exponential growth
		# base proportion max diff 100 fold
		icpt <- runif(q, -2.3, 2.3)
	}
	if (s==3) {
		# linear growth
		# base proportion max diff 100 fold
		icpt <- runif(q, 0.02, 2)
		g.m <- X[, -1] %*% t(f*b) 
		adj <- apply(g.m, 2, min)
		icpt[adj < 0] <- icpt[adj < 0] - adj[adj < 0] + 0.02
	}

	b <- cbind(icpt, f*b)
	
	for (i in 1:n){
		if (s==1 || s==2){
			# Exponential function
			g <- as.vector(exp(b %*% X[i, ]))
		} else if (s==3) {
			# Linear function
			g <- as.vector(b %*% X[i, ])
		} else if (s==4) {
			# atan function
			X.temp <- c(1, 0.25 * X[i, -1])
			g <- as.vector(atan(b %*% X.temp))
		}
		pi[i, ] <- g / sum(g)
		if (s==1) {
			# Exactly the same as we model the data
			theta <- 1 / (sum(g)+1)
		}
		if (theta == 0){
			Y[i, ] <- rmultinom(1, ct[i], pi[i, ])[, 1]
		} else {
			Y[i, ] <- simPop(J=1, n=ct[i], pi=pi[i, ], theta=theta)$data[1, ]
		}
	}
	cat("Finished!\n")
	return(list(X=X, Y=Y, b=b, theta=theta0, pi=pi, s=s, p=p, q=q, nr=nr,
					p.g=p.g, q.g=q.g, rho=rho, f=f))
}


#############################
#----------------------------------------------------------------------------------------------------
mysdr4dirmult <- function(Y,X,lamda.max0=NULL,n.grid=50){
  fy = cbind(rep(1,length(Y)),as.matrix(get_fy(Y,type='disc')))
  dm.obj <- myDirmultGrpGrid(X,fy,model="dirmult")
  opt = findOpt(dm.obj)
  b = dm.obj[[opt]]$b[,-1]
  q = dim(X)[2]
  Xnew = X; m = rowSums(X); Xnew[,q] = m-X[,q];
  R1 = as.matrix(Xnew)%*%b
  return(list(B = b, proj = cbind(R1,m)))
}







#----------------------------------------------------------------------------------------------------
getReduction <- function(X,b){
  q = dim(X)[2]
  Xnew = X; m = rowSums(X); Xnew[,q] = m-X[,q];
  R1 = as.matrix(Xnew)%*%b
  return(cbind(R1,m))
}


#----------------------------------------------------------------------------------------------------
get_fy <- function(Y,type=NULL,r=NULL){
  if (type=="disc"){fy <- get_fyZ(Y)}
  else {
    fy = matrix(0,nrow(Y),r)
    for (j in 1:r){
      aux = Y^j
      fy[,j] = aux - mean(aux)
    }
  }
  return(fy)
}

#----------------------------------------------------------------------------------------------------
findOpt <- function(obj){
  L = length(obj)
  control = -Inf
  for (i in 1:L){
    if (obj[[i]]$loglik > control){
      control = obj[[i]]$loglik
      opt = i
    }
  }
  return(opt)
}



##############################################################3

myDirmultSim <- function(nr=1000, s=2, q=40, q.g=4, fy = NULL,f=0.80, theta0=0.025){
  # Simulation strategy 
  # 	
  # Args:
  #		n: the number of samples
  #		nr: number of reads for each sample
  #   s: scenario, 2 - exponential growth, 3 - linear growth
  #   rho: the correlation between covariates
  #   p: number of covariates excluding the intercept
  #		q: number of species
  #   p.g: number of relevant nutrients
  #		q.g: number of relevant species
  #   f: controls the signal
  #		theta0: the dispersion parameter
  #
  # Returns:
  #		Y: count matrix, row - n samples, column - q species
  #		X: design matrix, with intercepts, n * (p+1)
  #		b: simulated coefficients q * (p+1)
  #		pi, theta: used by dirmult
  theta <- theta0
  cat("Generating data ...\n")
  
  fy = as.matrix(fy)
  p <- ncol(fy)
  p.g <- p
  n <- nrow(fy)
  X <- cbind(rep(1,n),fy)
             
  Y <- matrix(0, n, q)
  b <- matrix(0, q, p)
  pi <- matrix(0, n, q)
  ct <- sample(nr:(2*nr), n, rep=T)
  
  
  # coefficient with signs alternating
  st <- 0
  if (q.g != 1) {
    coef <- seq(0.6, 0.9, len=q.g) * c(1, -1)	
  } else {
    coef <- (0.6 + 0.9) / 2
  }
  
  coef.g <- seq(1.0, 1.0, len=p.g)
  
  for (i in 1:p.g) {
    # overlap two species
    # No overlaps for simplicity
    # q may be small so enable wrap-up
    b[(st:(st+q.g-1))%%q+1, i] <- coef.g[i] * coef[((i-1):(i+q.g-2))%%q.g+1]
    #st <- st+1  
  }
  
  if (s==1) {
    gs <- (1-theta) / theta
    # base proportion max diff 100 fold
    icpt <- runif(q, 0.02, 2)
    # The theta for each sample is different, but let them close to supplied value
    icpt <- gs * icpt / sum(icpt) 
    icpt <- log(icpt)		
  } 
  if (s==2) {
    # exponential growth
    # base proportion max diff 100 fold
    icpt <- sort(runif(q, -2.3, 2.3),decreasing = TRUE)
  }
  if (s==3) {
    # linear growth
    # base proportion max diff 100 fold
    icpt <- runif(q, 0.02, 2)
    g.m <- X[, -1] %*% t(f*b) 
    adj <- apply(g.m, 2, min)
    icpt[adj < 0] <- icpt[adj < 0] - adj[adj < 0] + 0.02
  }
  
  b <- cbind(icpt, f*b)
  
  for (i in 1:n){
    if (s==1 || s==2){
      # Exponential function
      g <- as.vector(exp(b %*% X[i, ]))
    } else if (s==3) {
      # Linear function
      g <- as.vector(b %*% X[i, ])
    } else if (s==4) {
      # atan function
      X.temp <- c(1, 0.25 * X[i, -1])
      g <- as.vector(atan(b %*% X.temp))
    }
    pi[i, ] <- g / sum(g)
    if (s==1) {
      # Exactly the same as we model the data
      theta <- 1 / (sum(g)+1)
    }
    if (theta == 0){
      Y[i, ] <- rmultinom(1, ct[i], pi[i, ])[, 1]
    } else {
      Y[i, ] <- simPop(J=1, n=ct[i], pi=pi[i, ], theta=theta)$data[1, ]
    }
  }
  cat("Finished!\n")
  return(list(X=X, Y=Y, b=b, theta=theta0, pi=pi, s=s, p=p, q=q, nr=nr,
              p.g=p.g, q.g=q.g, f=f))
}




#----------------------------------------------------------------------------------------------------
myrDirmult <- function(nr=1000, s=2, fy = NULL, b = NULL,f=0.80, theta0=0.025){
  theta <- theta0
  cat("Generating data ...\n")
  
  fy = as.matrix(fy)
  p <- ncol(fy)
  q <- nrow(b)
#  p.g <- p
  n <- nrow(fy)
  X <- cbind(rep(1,n),fy)
   
  Y <- matrix(0, n, q)
  pi <- matrix(0, n, q)
  ct <- sample(nr:(20000*nr), n, rep=TRUE)
  
  for (i in 1:n){
    if (s==1 || s==2){
      # Exponential function
      g <- as.vector(exp(b %*% X[i, ]))
    } else if (s==3) {
      # Linear function
      g <- as.vector(b %*% X[i, ])
    } else if (s==4) {
      # atan function
      X.temp <- c(1, 0.25 * X[i, -1])
      g <- as.vector(atan(b %*% X.temp))
    }
    pi[i, ] <- g / sum(g)
    if (s==1) {
      # Exactly the same as we model the data
      theta <- 1 / (sum(g)+1)
    }
    if (theta == 0){
      Y[i, ] <- rmultinom(1, ct[i], pi[i, ])[, 1]
    } else {
      Y[i, ] <- simPop(J=1, n=ct[i], pi=pi[i, ], theta=theta)$data[1, ]
    }
  }
  cat("Finished!\n")
  return(list(X=X, Y=Y, b=b, theta=theta0, pi=pi, s=s, p=p, q=q, nr=nr, f=f))
}

#----------------------------------------------------------------------------------------------------
myscatter <- function(x,y){
  if (dim(x)[2]>2){print('warning: only the first two columns will be plotted')}
  plot(x[,1],x[,2],col=Y+3,pch=16)
}  
  

#----------------------------------------------------------------------------------------------------
myangle1D <- function(x,y){
  x = x/sqrt(sum(x*x))
  y = y/sqrt(sum(y*y))
  return(acos(sum(x*y))*180/pi)
}


#----------------------------------------------------------------------------------------------------
get_fyZ <- function(Y){
  Y = as.integer(unlist(Y))
  n = length(Y)
  nclasses = length(unique(Y))
  Fy = matrix(0,n,nclasses-1)
  for (j in 1:(nclasses-1)){
    idx = which(Y==j)
    ni = length(idx)
    Fy[,j]=-ni/n
    Fy[idx,j] = 1-ni/n
  }
  return(Fy)
}


#----------------------------------------------------------------------------------------------------
cv.sdr4dirmult <- function(Y,counts,lamda.max0=NULL,n.grid=10,kfold=10,
                           nz=length(Y),
                           cc = 0.9){
  X = counts
  fy = cbind(rep(1,length(Y)),as.matrix(get_fy(Y,type='disc')))
  b0 = getInitialEstimate(X,fy)
  lambda.max = 10*TuningMax(X,fy,b0[,1],model="dirmult")
  n = length(Y)
  parts = sample(1:kfold,n,replace=TRUE)
  errores = matrix(NA,kfold,n.grid)
  for (fold in 1:kfold){
    test = parts==fold
    train = !test
    dm.obj <- myDirmultGrpGrid(X[train,],fy[train,],model="dirmult",lamda.max0 = lambda.max,n.grid = n.grid,nz=nz,cc=cc)
    kmax = min(n.grid,length(dm.obj))
    #lambda.range=NULL
    for (k in 1:kmax){
      b = dm.obj[[k]]$b[,-1]
     # lambda.range[k]=dm.obj[[k]]$lamda
      if (sum(b)!=0){
      Rtrain = getReduction(X[train,],b)
      Rtest = getReduction(X[test,],b)
      aux = cv.glmnet(Rtrain,as.matrix(Y[train]),family="multinomial",nlambda=20)
      yprob = predict(aux,Rtest,s=aux$lambda.min)
      yhat = apply(yprob,1,which.max)
      errores[fold,k] = mean(yhat!=Y[test])
      }
    }
    meanerr = apply(errores,2,mean,na.rm=TRUE)
    meanerr = meanerr[meanerr!=0]
    opt = which.min(meanerr)
    #lamda.range = lambda.max * 0.96^(0:(kmax-1))
    lamda.range <- logspace(log10(lambda.max),-6,n=n.grid)
    lambda.min = lamda.range[opt]
    return(list(lambda.min=lambda.min,lambda=lamda.range,cverror=meanerr))
  }
}



#----------------------------------------------------------------------------------------------------
cv.sdr4dirmult.lr <- function(Y,counts,lamda.max0=NULL,n.grid=10,kfold=10,
                           nz=length(Y),
                           cc = 1.0){
  X = counts
  fy = cbind(rep(1,length(Y)),as.matrix(get_fy(Y,type='disc')))
  b0 = getInitialEstimate(X,fy)
  lambda.max = TuningMax(X,fy,b0[,1],model="dirmult")
  
  n = length(Y)
  parts = sample(1:kfold,n,replace=TRUE)
  errores = matrix(NA,kfold,n.grid)
  for (fold in 1:kfold){
    test = parts==fold
    train = !test
    dm.obj <- myDirmultGrpGrid(X[train,],fy[train,],model="dirmult",lamda.max0 = lambda.max,n.grid = n.grid,nz=nz,cc=cc)
    kmax = min(n.grid,length(dm.obj))
    for (k in 1:kmax){
      b = dm.obj[[k]]$b[,-1]    
      if (sum(b)!=0){
        Rtrain = getReduction(X[train,],b)
        Rtest = getReduction(X[test,],b)
        aux = cv.glmnet(Rtrain,as.matrix(Y[train]),family="binomial",nlambda=20)
        yprob = predict(aux,Rtest,s=aux$lambda.min)
        yhat = apply(yprob,1,which.max)
        errores[fold,k] = mean(yhat!=Y[test])
      }
    }
    meanerr = apply(errores,2,mean,na.rm=TRUE)
    meanerr = meanerr[meanerr!=0]
    opt = which.min(meanerr)
    lamda.range = lambda.max * 0.96^(0:(kmax-1))
    lambda.min = lamda.range[opt]
    return(list(lambda.min=lambda.min,lambda=lamda.range,cverror=meanerr))
  }
}


#----------------------------------------------------------------------------------------------------
cv.sdr4dirmult.lin <- function(Y,counts,lamda.max0=NULL,n.grid=50,kfold=5,cc=.20,r=2, nz = length(Y) ){
  X = counts
  fy = cbind(rep(1,length(Y)),as.matrix(get_fy(Y,type="cont",r=r)))
  b0 = getInitialEstimate(X,fy)
  lambda.max = TuningMax(X,fy,b0[,1],model="dirmult")
  
  n = length(Y)
  parts = sample(1:kfold,n,replace=TRUE)
  errores = matrix(0,kfold,n.grid)
  for (fold in 1:kfold){
    test = parts==fold
    train = !test
    dm.obj <- myDirmultGrpGrid(X[train,],fy[train,],model="dirmult",lamda.max0 = lambda.max,n.grid = n.grid,cc=cc,nz=nz)
    kmax = min(n.grid,length(dm.obj))
    for (k in 1:kmax){
      b = dm.obj[[k]]$b[,-1]    
      Rtrain = getReduction(X[train,],b)
      Rtest = getReduction(X[test,],b)
      aux = glmnet(cbind(rep(1,sum(train)),Rtrain),as.matrix(Y[train]),family="gaussian",lambda=0.001)
      yhat = predict(aux,cbind(rep(1,sum(test)),Rtest),s=0.0)
      errores[fold,k] = mean(mynorm(yhat-Y[test])^2)
    }
    meanerr = apply(errores,2,mean)
    meanerr = meanerr[meanerr!=0]
    opt = which.min(meanerr)
    lamda.range = lambda.max * 0.96^(0:(kmax-1))
    lambda.min = lamda.range[opt]
    return(list(lambda.min=lambda.min,lambda=lamda.range,cverror=meanerr))
  }
}


#----------------------------------------------------------------------------------------------------
getInitialEstimate <- function(Y,X,model="dirmult",initscalar=10){
  p <- ncol(X)
  q <- ncol(Y)
  if (sum(X[, 1] != 1) != 0) {
    warning("No intercepts in X! Intercepts added\n")
    X <- cbind(rep(1, nrow(X)), X)
  }
  if (nrow(Y) != nrow(X)) stop("Y and X have different sample size!\n")
  if (sum(round(Y) != Y)) stop("Y should contain counts!\n")
  if (model == "dir") {	
    Y <- (Y + 0.5) / rowSums(Y)
  }	
  # if (model == "dirmult") {
  #   cat("Sparse Dirichlet-Multinomial Regression\n")
  # }
  # if (model == "dir") {
  #   cat("Sparse Dirichlet Regression\n")
  # }
  # if (model == "mult") {
  #   cat("Sparse Multinomial Regression\n")
  # }
  # cat("Initial MLE of the coefficient matrix ...\n")
  b00 <- matrix(0, q, p,dimnames = list(dimnames(Y)[[2]],NULL))
  if (model == "dirmult" ) {
    aux = dirmult(Y, initscalar = initscalar, trace=FALSE)
    aux2 = log(aux$pi*aux$theta)
    b00[names(aux2),1] <- aux2
  } 
  if (model == "mult" | model == "dir") {
    f1 <- function(b) {
      b00[, 1] <- b
      -Loglik(Y, X, b00, model)
    }	
    b00[, 1] <- nlm(f1, rep(0, q))$estimate
  } 
  return(b00)
}


sdr4dirmult_v2 <- function(Y,counts,lamda=0,cc=0){
  X = counts
  fy = cbind(rep(1,length(Y)),as.matrix(get_fy(Y,type='disc')))
  b0 = getInitialEstimate(X,fy)
  dm.obj <- myFullGrpLasso(X, fy, b0, lambdaInit=0.0,lambda = lamda, ccInit=0.0, cc=cc)
  b = dm.obj$b[,-1]
  q = dim(X)[2]
  Xnew = X; m = rowSums(X); Xnew[,q] = m-X[,q];
  R1 = as.matrix(Xnew)%*%b
  return(list(B = b, proj = cbind(R1,m)))
}


sdr4dirmult_v2cont <- function(Y,counts,lambda=0,cc=0,r=2){
  X = counts
  fy = cbind(rep(1,length(Y)),as.matrix(get_fy(Y,type='cont',r=r)))
  b0 = getInitialEstimate(X,fy)
  dm.obj <- myDirmultGrp(X,fy,b0,cc=cc,lamda=lambda,model="dirmult")
  b = dm.obj$b[,-1]
  q = dim(X)[2]
  Xnew = X; m = rowSums(X); Xnew[,q] = m-X[,q];
  R1 = as.matrix(Xnew)%*%b
  return(list(B = b, proj = cbind(R1,m)))
}


delnan <-function(x){
  idx = which((is.nan(x)))
  x[idx]=0
  return(x)
}

logComp <- function(X,base=ncol(X)){
  N = nrow(X)
  eps = 1e-16
  X = X+eps
  Y = matrix(NA,N,ncol(X))
  for (j in 1:ncol(X)){
    Y[,j] = log(X[,j]/X[,base])
  }
  Y = Y[,-base]
  return(Y)
}

# pfc4comp <- function(Y,X,dim=2,base=ncol(X)){
#   Z = logComp(X,base)
#   fy = get_fyZ(Y)
#   out = pfc(Z,Y,fy=fy,numdir=dim,structure="aniso")
#   return(out)
# }


myWgrplasso <- function(Y,X,b0,lamda=0,cc=1.0,r=0.5,model="dirmult",doweight=TRUE,
                        alpha0   = 1,
                        delta    = 0.5,
                        sigma    = 0.1,
                        cstar    = 0.001,
                        tol     = 1e-4,
                        iter.lim = 100) {
  
  b0 <- as.matrix(b0)		
  b00 <- b0
  n <- nrow(X)
  p <- ncol(b0)
  q <- nrow(b0)
  if (doweight){
      weights = apply(Y,2,mean)^r
  }else {weights = rep(1,q)}
  b1 <- b0
  if (lamda > 0){
    c1 <- sqrt(p-1) * lamda * cc; # * cc # so c1, c2 are on the same level
    c2 = (1-cc) * lamda
    iter2 <- 0
    
    repeat {
      b0 <- b1
      h <- H(Y, X, b0, model)
      s <- S(Y, X, b0, model)
      h1 <- -max(-h[, 1], cstar)
      d1 <- -s[, 1] / h1
      if (sum(d1 != 0) != 0){
        Delta <- - sum(d1 * s[, 1])
        alpha <- alpha0
        alpha.d1 <- alpha * d1
        while (ObjFunc(b1[, 1] + alpha.d1, 1, b1, Y, X, model=model) - 
               ObjFunc(b1[, 1], 1, b1, Y, X, model=model) > alpha*sigma*Delta) {
          alpha <- alpha * delta
          alpha.d1 <- alpha * d1
          if (max(abs(alpha.d1)) < 0.1*tol){alpha.d1 <- 0; break}
        }
        b1[, 1] <- b1[, 1] + alpha.d1
      }
      
      di <- matrix(0,p,1)
      for (i in 1:q) {
        hi <- -max(-h[i,-1], cstar)
        diff.v <- s[i,-1]- hi * b1[i,-1]
        ind <- (abs(diff.v) <= c2*weights[i])
        ind = c(FALSE,ind)
        di[ind] <-  -b1[i,ind] ; di[1]=0	
        if (sum(ind) != (p-1)) {
          ind <- !ind				
          ind[1] = FALSE
          si.temp <- s[i,ind] - c2*weights[i]*(sign(s[i,] - hi*b1[i,])[ind])				
          diff.v <- si.temp - hi*b1[i, ind] #	dimension changed
          diff.n <- sqrt(sum(diff.v^2))
          if (diff.n <= c1*weights[i]){				
            di[ind]  <- -b1[i, ind]; 
          } else {	
            di[ind] <- -(si.temp - c1*weights[i]*diff.v/diff.n) / hi
          }
          di[1]=0
        }
        
        if (sum(di != 0) != 0){
          Delta <- - sum(di * s[i,]) + 
            c1*weights[i] * (sqrt(sum((b1[i,-1] + di[-1])^2)) - sqrt(sum(b1[i,-1]^2))) +
            c2*weights[i] * sum(abs(b1[i,-1] + di[-1]) - abs(b1[i,-1]))
          alpha <- alpha0
          alpha.di <- alpha * di
          b2 = b1[i,] + alpha.di
          resta = ObjFunc2(b2, i, b1, Y, X, c1*weights[i], c2*weights[i], model) - ObjFunc2(b1[i,], i, b1, Y, X, c1*weights[i], c2*weights[i], model)
          if (is.nan(resta)){print('_____________SALIENDO POR nan________________'); print(iter2); break}
          bound = alpha*sigma*Delta
          while (resta > bound) {
            alpha <- alpha * delta
            alpha.di<- alpha * di
            if (max(abs(alpha.di)) < 0.1*tol){ 
              alpha.di <- 0; 
              break 
            }
            b2 = b1[i,] + alpha.di
            resta = ObjFunc2(b2, i, b1, Y, X, c1*weights[i], c2*weights[i], model) - ObjFunc2(b1[i,], i, b1, Y, X, c1*weights[i], c2*weights[i], model)
            if (is.nan(resta)){print('_____________SALIENDO POR nan________________'); print(iter2); break}
          }
          b1[i,] <- b1[i,] + alpha.di
        }
      }
      iter2 <- iter2 + 1
      if (iter2 > iter.lim) {
        break
      }
      if (max(abs(b1-b0)) <= tol) break
    }
  }
  lik <- n * Loglik(Y, X, b1, model)
  lik0 <- n * Loglik(Y, X, b00, model)
  return(list(b=b1, lamda=lamda, cc=cc, loglik=lik, loglik0 = lik0))
}


mynorm <- function(v){
  sqrt(sum(v*v))
}


myInitGrpLasso <- function(Y, X, b0, lamda=0, cc=0, model="dirmult",
                         alpha0   = 1,
                         delta    = 0.5,
                         sigma    = 0.1,
                         cstar    = 0.001,
                         tol     = 1e-4,
                         iter.lim = 20) {
  # Minimize objective function: -Loglik +  group lasso + lasso
  # 	
  b0 <- as.matrix(b0)		
  b00 <- b0
  n <- nrow(X)
  p <- ncol(b0)
  q <- nrow(b0)
  
  # PRIMERA ETAPA: OBTENER ESTIMADOR INICIAL sin penalizacion
  c1 <- sqrt(q) * lamda * cc # so c1, c2 are on the same level
  c2 <- lamda * (1-cc)
  
  print(c1)
  print(c2)

  # check inputs
  if (ncol(Y) != q || ncol(X) != p) stop("Dimension does not match!\n")
  if (sum(X[, 1] != 1) != 0) {
    warning("No intercepts in X! Intercepts added\n")
    X <- cbind(rep(1, nrow(X)), X)
  }
  if (nrow(Y) != nrow(X)) stop("Y and X have different sample sizes!\n")	
  if (model == "dir") {	
    if (max(Y) > 1) {
      Y <- (Y + 0.5) / rowSums(Y)
    } else {
      if (min(Y) == 0) {
        Y[Y==0] <- min(Y[Y!=0])/10
      }
    }
  }
  
  
  #-------------- MAIN LOOP--------------------------------
  b1 <- b0
  iter <- 0
  repeat {
    b0 <- b1
    h <- H(Y, X, b0, model)
    s <- S(Y, X, b0, model)
    h1 <- -max(-h[, 1], cstar) 
    d1 <- -s[, 1] / h1         
    if (sum(d1 != 0) != 0){
      Delta <- - sum(d1 * s[, 1])
      alpha <- alpha0
      alpha.d1 <- alpha * d1
      while (ObjFunc(b1[, 1] + alpha.d1, 1, b1, Y, X, model=model) - 
             ObjFunc(b1[, 1], 1, b1, Y, X, model=model) > alpha*sigma*Delta) {
        alpha <- alpha * delta
        alpha.d1 <- alpha * d1
        if (max(abs(alpha.d1)) < 0.1*tol){alpha.d1 <- 0; break}
      }
      b1[, 1] <- b1[, 1] + alpha.d1
    }#-------------------------------------------------------end for the intercepts

    # ---------- Minimize with respect to the penalized groups
    di <- numeric(q)
    if (p != 1){
      for (i in 2:p) {
        hi <- -max(-h[, i], cstar)
        diff.v <- s[, i]- hi * b1[, i]
        ind <- (abs(diff.v) <= c2)
        di[ind] <-  -b1[ind, i]	
        if (sum(ind) != q) {
          ind <- !ind				
          si.temp <- s[ind, i] - c2*(sign(s[, i] - hi*b1[, i])[ind])				
          diff.v <- si.temp - hi*b1[ind, i]
          diff.n <- sqrt(sum(diff.v^2))
          if (diff.n <= c1){				
            di[ind]  <- -b1[ind, i]			
          } else {	
            di[ind] <- -(si.temp - c1*diff.v/diff.n) / hi
          }
        }
        
        if (sum(di != 0) != 0){
          Delta <- - sum(di * s[, i]) + 
            c1 * (sqrt(sum((b1[, i] + di)^2)) - sqrt(sum(b1[, i]^2))) +
            c2 * sum(abs(b1[, i] + di) - abs(b1[, i]))
          alpha <- alpha0
          alpha.di <- alpha * di
          while (ObjFunc(b1[, i] + alpha.di, i, b1, Y, X, c1, c2, model) - 
                 ObjFunc(b1[, i], i, b1, Y, X, c1, c2, model) > alpha*sigma*Delta) {
            alpha <- alpha * delta
            alpha.di<- alpha * di
            # This step is used to eliminate those small non-zero coefficients
            if (max(abs(alpha.di)) < 0.1*tol){ alpha.di <- 0; break }
          }
          b1[, i] <- b1[, i] + alpha.di
        }
      }
    }
    iter <- iter + 1
    if (iter > iter.lim) {
      break
    }
    if (max(abs(b1-b0)) <= tol) break
  }
  
  lik <- n * Loglik(Y, X, b1, model)
  return(list(b=b1, lamda=lamda, cc=cc, iter=iter, loglik=lik))
}



myFullGrpLasso <- function(Y,X,b0,lambdaInit=0.0,lambda=0,ccInit=0.0,cc=1.0,r=0.5,model="dirmult",doweight=TRUE,
                        alpha0   = 1,
                        delta    = 0.5,
                        sigma    = 0.1,
                        cstar    = 0.001,
                        tol     = 1e-4,
                        iter.lim = 500) {
  #
  step1 = myInitGrpLasso(Y,X,b0,lamda=lambdaInit,cc=ccInit,model=model)
  step2 = myWgrplasso(Y,X,step1$b,lamda=lambda,cc=cc,r=r,model=model,doweight=doweight)
  return(step2)
}  




Ini.Logratios <- function(base, W) {
  W.m <- as.matrix(W)
  N <- nrow(W.m)
  Q <- ncol(W.m)
  M = apply(W.m, 1, sum)
  Psedo_Z.m <- matrix(0, N, Q)
  Y.m <- matrix(0, N, Q - 1)
  for (i in 1:N) {
    zeros <- which(W.m[i, ] == 0)
    nzeros <- which(W.m[i, ] != 0)
    Psedo_Z.m[i, zeros] <- (W.m[i, zeros] + 0.05)/(M[i] + 
                                                     0.05 * length(zeros))
    Psedo_Z.m[i, nzeros] <- (W.m[i, nzeros])/(M[i] + 
                                                0.05 * length(zeros))
    Y.m[i, ] <- log(Psedo_Z.m[i, -base]/Psedo_Z.m[i, 
                                                  base])
  }
  attr(Y.m, "center") = apply(Y.m, 2, mean)
  attr(Y.m, "base") = paste("the", base, "th taxa")
  Y.m
}

myprojdist <- function(u,v){
  u = as.matrix(u)
  v = as.matrix(v)
  pu = u%*%solve(t(u)%*%u)%*%t(u)
  pv = v%*%solve(t(v)%*%v)%*%t(v)
  er = pu-pv
  out = norm(er,'f')/norm(pu,'f')
  return(out)
}

LogisticT <- function(base, b) {
  if (is.vector(b)) {
    b.full <- rep(0, length(b) + 1)
    b.full[-base] <- b
    b.simplex <- exp(b.full)/sum(exp(b.full))
  }
  if (is.matrix(b)) {
    b.full <- matrix(0, nrow(b), ncol(b) + 1)
    b.full[, -base] <- b
    b.simplex <- t(apply(b.full, 1, function(x) {
      exp(x)/sum(exp(x))
    }))
  }
  b.simplex
}



PGM.invreg <- function(X,Z,epsilon=1e-1){
  p=dim(X)[2]
  n=dim(X)[1]
  q=dim(Z)[2]+1
  d=p*(p+1)/2
  stopifnot(dim(Z)[1]==n)
  # constants
  eyep = diag(rep(1,p))
  A=matrix(0,p*p,p)
  for (i in 0:(p-1)){A[i*p+i+1,i+1]=1}
  Dp=duplication.matrix(p)
  DpTA=t(Dp)%*%A
  # vars
  K1  = matrix(NA,p,p)
  K2  = matrix(NA,d,p)
  h1  = matrix(NA,p,1)
  h2  = matrix(NA,d,1)
  KhT = matrix(NA,1,p)
  # expectations
  EUpsK11Ups = matrix(0,p*q,p*q)# como KK11=Ip
  EK22       = matrix(0,d,d)
  EUpsK12    = matrix(0,q*p,d)
  EK2h       = matrix(0,d,1)
  EUpsK1h    = matrix(0,q*p,1)
  for (i in 1:n){
    x = t(X[i,,drop=F])+epsilon
    z = rbind(1,t(Z[i,,drop=F]))
    u      = sqrt(x)
    u.inv  =1/u
    u.inv3 =u.inv^3
    K1  = diag(u.inv[,1]/2)
    K2  = t(Dp)%*%kronecker(u,diag(u.inv[,1]))
    KhT = -digamma(x+1)
    h1  = -u.inv3/4
    h2  = (-t(Dp)%*%kronecker(u,u.inv3)+DpTA%*%(1/x))/2
    Ups = kronecker(t(z),eyep)
    EUpsK11Ups  = EUpsK11Ups  + t(Ups)%*%K1%*%t(K1)%*%Ups  
    EK22        = EK22        + K2%*%t(K2)    
    EUpsK12     = EUpsK12     + t(Ups)%*%K1%*%t(K2)  
    EK2h        = EK2h        + K2%*%KhT + h2
    EUpsK1h     = EUpsK1h     + t(Ups)%*%(K1%*%KhT+h1)
  }
  EUpsK11Ups =EUpsK11Ups/n
  EK22inv    =ginv(EK22/n)
  EUpsK12    =EUpsK12/n
  EK2h       =EK2h/n
  EUpsK1h    =EUpsK1h/n
  #closed form 
  tmp=EUpsK12%*%EK22inv
  A=EUpsK11Ups-tmp%*%t(EUpsK12)
  B=tmp%*%EK2h-EUpsK1h
  b     = ginv(A)%*%B
  theta2=-EK22inv%*%(t(EUpsK12)%*%b+EK2h)
  Theta2= matrix(Dp%*%theta2,p,p)
  B     = matrix(b,p,q)
  return(list("B"=B,"Theta2"=Theta2))
}
