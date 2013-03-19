SegAnnotBases <- structure(function
### Annotation-aware segmentation via dynamic programming. We assume a
### complete 0/1 annotation of the signal, so you just need to specify
### where the 1-annotated regions occur, in terms of base pairs. We
### first convert starts and ends from base pairs to indices, then we
### use the SegAnnot function.
(y,
### The vector of observations to segment.
 base,
### Position of each observation in base pairs.
 starts,
### Starts of the 1-annotated regions in base pairs.
 ends
### Ends of the 1-annotated regions in base pairs.
 ){
  stopifnot(length(y)==length(base))
  stopifnot(is.numeric(y))
  stopifnot(is.integer(base))
  stopifnot(length(starts)==length(ends))
  stopifnot(is.integer(starts))
  stopifnot(is.integer(ends))
  ord <- order(base)
  y <- y[ord]
  base <- base[ord]
  m <- sapply(starts,function(x){
    i <- which(base < x)
    i[length(i)]
  })
  M <- sapply(ends,function(x){
    i <- which(base > x)
    i[1]
  })
  result <- SegAnnot(y, m, M)
  break.i <- result$change[-c(1,length(result$change))]
  break.base <- 
    floor( (base[break.i] + base[break.i+1])/2 )
  break.draw <- break.base+1/2
  first.base <- c(base[1]-1/2,break.draw)
  last.base <- c(break.draw,base[length(base)]+1/2)
  result$seg.df <-
    data.frame(first.base,last.base,
               first.i=c(1,break.i+1),last.i=c(break.i,length(y)),
               mean=result$smt[result$change[-1]])
  result$break.df <- data.frame(base=break.base)
  result$signal.df <- data.frame(base,signal=y,smooth=result$smt)
  result$ann.df <-
    data.frame(first.base=starts,last.base=ends,
               first.i=m,last.i=M,annotation="1breakpoint")
  result
### List of results describing the segmentation model. In addition to
### the results of the SegAnnot function, there are data.frames
### seg.df, break.df, signal.df, and ann.df which can be easily
### plotted for model interpretation.
},ex=function(){
  data(profiles,package="SegAnnot")
  pro <- profiles$low
  fit <- SegAnnotBases(pro$pro$log,pro$pro$pos,pro$ann$min,pro$ann$max)
  library(ggplot2)
  p <- ggplot()+
    geom_tallrect(aes(xmin=first.base/1e6,xmax=last.base/1e6,fill=annotation),
                  data=fit$ann.df)+
    geom_point(aes(base/1e6,signal),data=fit$signal.df)+
    geom_segment(aes(first.base/1e6,mean,xend=last.base/1e6,yend=mean),
                 data=fit$seg.df,
                 colour=signal.colors[["estimate"]],lwd=1)+
    geom_vline(aes(xintercept=base/1e6),data=fit$break.df,
               colour=signal.colors[["estimate"]],lwd=1,linetype="dashed")+
    scale_fill_manual(values=breakpoint.colors)
  print(p)
})

SegAnnot <- structure(function
### Annotation-aware segmentation via dynamic programming. We assume a
### complete 0/1 annotation of the signal, so you just need to specify
### where the 1-annotated regions occur, in terms of indices. The
### solver gives you the solution y of min_y ||y-x||^2 such that there
### is exactly 1 break in each of the annotated regions, and no breaks
### elsewhere. See also the SegAnnotBases function, for which the
### annotations are specified in terms of base pairs.
(x,
### Numeric vector: the signal to segment.
 sR,
### Starts of the 1-annotated regions.
 eR
### Ends of the 1-annotated regions.
 ){
  stopifnot(length(sR)==length(eR))
  stopifnot(is.numeric(x))
  nMax <- length(x)
  stopifnot(nMax > 1)
  for(i in list(sR,eR)){
    stopifnot(is.integer(i))
    stopifnot(all(i >= 1))
    stopifnot(all(i <= nMax))
  }
  stopifnot(all(eR > sR))
  ## sort them first, in case they are not specified in increasing
  ## order.
  ord <- order(sR)
  sR <- sR[ord]
  eR <- eR[ord]
  sR = as.integer(sR-1)
  eR = as.integer(eR-1)
  sR = c(sR, nMax-1)
  eR = c(eR, nMax-1)
  pMax = length(eR)
  idPath = integer(pMax);
  cost= 0.0
  result <- .C('bridge_FindOptimalSegmentations',
               as.double(x), as.integer(sR), as.integer(eR), 
               as.integer(nMax), as.integer(pMax), as.integer(idPath),
               as.double(cost))
  names(result) <- c("x", "sR", "eR", "nMax", "pMax", "idPath", "cost")
  result$change <- c(0, sR+1)+ c(result$idPath, 0)
  result$smt <- smoothedSignal(x, result$change, method=mean)
  result$cost <- result$cost + sum(x^2)
  result
### List describing the segmentation model.
},ex = function(){
  set.seed(1)
  x <- rnorm(100000)+	rep(c(1, 2), c(600, 1e5 - 600))

  sR = as.integer(c(500, 2000))
  eR = as.integer(c(1900, 5000))
  result1 <- SegAnnot(x, sR, eR)
  which(diff(result1$smt)!=0)
  sR = as.integer(c(1000, 2000))
  eR = as.integer(c(2100, 5000))
  result2 <- SegAnnot(x, sR, eR)
  which(diff(result2$smt)!=0)
})

SegAnnotBasesC <- structure(function
### C implementation of mapping bases to indices.
(y,
### The vector of observations to segment.
 base,
### Position of each observation in base pairs.
 starts,
### Starts of the 1-annotated regions in base pairs.
 ends
### Ends of the 1-annotated regions in base pairs.
 ){
  stopifnot(length(y)==length(base))
  stopifnot(is.numeric(y))
  stopifnot(is.integer(base))
  stopifnot(length(starts)==length(ends))
  stopifnot(is.integer(starts))
  stopifnot(is.integer(ends))
  ## sort the bases/signal in case they are not specified in
  ## increasing order.
  ord <- order(base)
  y <- y[ord]
  base <- base[ord]
  nMax <- length(y)
  stopifnot(nMax > 1)
  for(b in list(starts, ends)){
    stopifnot(is.integer(b))
    stopifnot(all(b >= 1))
  }
  stopifnot(all(ends > starts))
  ## sort the regions in case they are not specified in increasing
  ## order.
  ord <- order(starts)
  starts <- starts[ord]
  ends <- ends[ord]
  n.regions <- length(ends)
  segStart <- integer(n.regions+1)
  sR <- integer(n.regions+1)
  eR <- integer(n.regions+1)
  cost <- 0
  status <- -1
  result <- .C('bridge_bases',
               as.double(y), as.integer(base),
               as.integer(starts), as.integer(ends), 
               as.integer(sR), as.integer(eR), 
               as.integer(nMax), as.integer(n.regions),
               as.integer(segStart), as.integer(status),
               as.double(cost))
  names(result) <- c("x", "base",
                     "starts", "ends",
                     "sR", "eR",
                     "nMax", "n.regions",
                     "segStart", "status",
                     "cost")
  if(result$status != 0){
    stop("error code ", result$status)
  }
  result$cost <- result$cost + sum(y^2)
  result$change <- c(result$segStart, nMax)
  result$smt <- smoothedSignal(y, result$change, method=mean)
  break.i <- result$change[-c(1,length(result$change))]
  break.base <- 
    floor( (base[break.i] + base[break.i+1])/2 )
  break.draw <- break.base+1/2
  first.base <- c(base[1]-1/2,break.draw)
  last.base <- c(break.draw,base[length(base)]+1/2)
  result$seg.df <-
    data.frame(first.base,last.base,
               first.i=c(1,break.i+1),last.i=c(break.i,length(y)),
               mean=result$smt[result$change[-1]])
  result$break.df <- data.frame(base=break.base)
  result$signal.df <- data.frame(base,signal=y,smooth=result$smt)
  result$ann.df <-
    data.frame(first.base=starts,last.base=ends,
               first.i=result$sR[-length(result$sR)],
               last.i=result$eR[-length(result$eR)],
               annotation="1breakpoint")
  result
},ex=function(){
  library(SegAnnot)
  data(profiles,package="SegAnnot")
  if(interactive()){
    for(info in profiles){
      ann <- subset(info$ann, annotation=="1breakpoint")
      pro <- info$pro
      fit <- SegAnnotBases(pro$log, pro$pos, ann$min, ann$max)
      fitc <- SegAnnotBasesC(pro$log, pro$pos, ann$min, ann$max)
      for(must.be.equal in c("sR","eR","change","cost")){
        stopifnot(all(fit[[must.be.equal]] == fitc[[must.be.equal]]))
      }
    }
  }
})


### Calculate the vector of smoothed observations.
smoothedSignal <- function(x, saut, method=mean){
  ctSeg <- diff(saut)
  segment <- rep(1:(length(saut)-1), ctSeg)
  smt <- aggregate(x, by=list(segment), FUN=method)[, 2]
  smtSignal <- rep(smt, ctSeg)
}


