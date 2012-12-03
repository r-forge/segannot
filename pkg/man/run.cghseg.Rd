\name{run.cghseg}
\alias{run.cghseg}
\title{run cghseg}
\description{Estimate the least squares model for a noisy signal. We use the
cghseg package, which implements the pruned dynamic programming
method of Rigaill (2010) to find, for all k=1,...,maxSegments:
argmin_{x has k segments} ||Y-x||^2_2.}
\usage{run.cghseg(Y, base = seq_along(Y), maxSegments = 20)}
\arguments{
  \item{Y}{Numeric vector of the noisy signal to segment.}
  \item{base}{Integer vector of bases where Y is sampled.}
  \item{maxSegments}{Maximum number of segments to consider.}
}

\value{List containing the solutions. The "segments" element is a
data.frame that describes the segmentation model, with 1 line for
each segment.}

\author{Guillem Rigaill, Toby Dylan Hocking}





