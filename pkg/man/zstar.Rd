\name{zstar}
\alias{zstar}
\title{Calculation of lambda}
\description{Given a set of optimal cost_k from k=1 segments to K segments,
calculate an exact representation of the model selection function
z^*(L)=argmin_k exp(L)*k + cost_k.}
\usage{zstar(cost)}
\arguments{
  \item{cost}{numeric vector: optimal costs.}
}

\value{Data.frame with columns min.lambda, max.lambda, segments, max.L,
min.L, size, an exact representation of the z^* function, where
min.L <= z^*(L) = segments <= max.L holds for each row.}

\author{Guillem Rigaill, Toby Dylan Hocking}





