\name{SegAnnot}
\alias{SegAnnot}
\title{SegAnnot}
\description{Annotation-aware segmentation via dynamic programming. We assume a
complete 0/1 annotation of the signal, so you just need to specify
where the 1-annotated regions occur, in terms of indices. The
solver gives you the solution y of min_y ||y-x||^2 such that there
is exactly 1 break in each of the annotated regions, and no breaks
elsewhere. See also the SegAnnotBases function, for which the
annotations are specified in terms of base pairs.}
\usage{
SegAnnot(x,
sR,
eR)
}
\arguments{
  \item{x}{Numeric vector: the signal to segment.}
  \item{sR}{Starts of the 1-annotated regions.}
  \item{eR}{Ends of the 1-annotated regions.}
}

\value{List describing the segmentation model.}

\author{Guillem Rigaill, Toby Dylan Hocking}




\examples{
set.seed(1)
x <- rnorm(100000)+	rep(c(1, 2), c(600, 1e5 - 600))

sR = c(500, 2000)
eR = c(1900, 5000)
result1 <- SegAnnot(x, sR, eR)
which(diff(result1$smt)!=0)
sR = c(1000, 2000)
eR = c(2100, 5000)
result2 <- SegAnnot(x, sR, eR)
which(diff(result2$smt)!=0)
}
