zstar <- function # Calculation of lambda
### Given a set of optimal cost_k from k=1 segments to K segments,
### calculate an exact representation of the model selection function
### z^*(L)=argmin_k exp(L)*k + cost_k.
(cost
### numeric vector: optimal costs.
 ){
  Kmax = length(cost)
  Kcurrent = Kmax
  Lcurrent = 0

  vK = Kmax
  vL = 0
  i <- 2
  while(Kcurrent > 1){
    smallerK <- 1:(Kcurrent-1)
    ## DO NOT DIVIDE BY N, then we can use the log.n feature and the
    ## L1 norm will drive it to zero, not to 1.
    cost.term <- (cost[Kcurrent] - cost[smallerK]) ##/ n
    lambdaTransition <-  cost.term / ( smallerK - Kcurrent)
    Kcurrent <- which.min(lambdaTransition)
    Lcurrent <- min(lambdaTransition)
    vL[i] <- Lcurrent
    vK[i] <- Kcurrent
    i <- i+1
  }
  ## vL[i] stores the smallest lambda such that vK[i] segments is
  ## optimal. i
  intervals <- 
    data.frame(min.lambda=vL,
               max.lambda=c(vL[-1],Inf),
               segments=vK)
  within(intervals,{
    max.L <- log(max.lambda)
    min.L <- log(min.lambda)
    size <- max.L-min.L
  })
### Data.frame with columns min.lambda, max.lambda, segments, max.L,
### min.L, size, an exact representation of the z^* function, where
### min.L <= z^*(L) = segments <= max.L holds for each row.
}

run.cghseg <- function
### Estimate the least squares model for a noisy signal. We use the
### cghseg package, which implements the pruned dynamic programming
### method of Rigaill (2010) to find, for all k=1,...,maxSegments:
### argmin_{x has k segments} ||Y-x||^2_2.
(Y,
### Numeric vector of the noisy signal to segment.
 base=seq_along(Y),
### Integer vector of bases where Y is sampled.
 maxSegments=20
### Maximum number of segments to consider.
 ){
  stopifnot(length(Y)==length(base))
  n <- length(Y)
  kmax <- min(maxSegments,n)#k is the number of SEGMENTS not BREAKPOINTS
  result <- cghseg:::segmeanCO(Y,kmax)
  result$zstar <- zstar(result$J)
  result$segments <- data.frame()
  result$breaks <- list()
  result$break.df <- data.frame()
  for(k in 1:kmax){
    ends <- result$t.est[k, 1:k]
    starts <- c(1,ends[-length(ends)]+1)
    signal <- rep(NA,k)
    for(seg.i in seq_along(starts)){
      start <- starts[seg.i]
      end <- ends[seg.i]
      signal[seg.i] <- mean(Y[start:end])
    }
    first.base <- base[starts]
    last.base <- base[ends]
    breaks <- floor((first.base[-1]+last.base[-k])/2)+1/2
    modelSegs <- data.frame(first.index=starts,last.index=ends,
                            first.base=c(first.base[1]-1/2,breaks),
                            last.base=c(breaks,last.base[k]+1/2),
                            mean=signal,segments=k)

    result$breaks[[k]] <- as.integer(breaks-1/2)

    if(k>1){
      result$break.df <- rbind(result$break.df,{
        data.frame(base=breaks,segments=k)
      })
    }

    result$segments <- rbind(result$segments,modelSegs)
  }
  result
### List containing the solutions. The "segments" element is a
### data.frame that describes the segmentation model, with 1 line for
### each segment.
}
