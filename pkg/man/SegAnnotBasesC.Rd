\name{SegAnnotBasesC}
\alias{SegAnnotBasesC}
\title{SegAnnotBasesC}
\description{C implementation of mapping bases to indices.}
\usage{SegAnnotBasesC(y, base, starts, ends)}
\arguments{
  \item{y}{The vector of observations to segment.}
  \item{base}{Position of each observation in base pairs.}
  \item{starts}{Starts of the 1-annotated regions in base pairs.}
  \item{ends}{Ends of the 1-annotated regions in base pairs.}
}



\author{Guillem Rigaill, Toby Dylan Hocking}




\examples{
library(SegAnnot)
data(profiles,package="SegAnnot")
for(info in profiles){
  ann <- subset(info$ann, annotation=="1breakpoint")
  pro <- info$pro
  fit <- SegAnnotBases(pro$log, pro$pos, ann$min, ann$max)
  fitc <- SegAnnotBasesC(pro$log, pro$pos, ann$min, ann$max)
  for(must.be.equal in c("sR","eR","change")){
    stopifnot(all(fit[[must.be.equal]] == fitc[[must.be.equal]]))
  }
}
}
