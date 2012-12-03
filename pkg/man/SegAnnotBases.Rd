\name{SegAnnotBases}
\alias{SegAnnotBases}
\title{SegAnnotBases}
\description{Annotation-aware segmentation via dynamic programming. We assume a
complete 0/1 annotation of the signal, so you just need to specify
where the 1-annotated regions occur, in terms of base pairs. We
first convert starts and ends from base pairs to indices, then we
use the SegAnnot function.}
\usage{SegAnnotBases(y, base, starts, ends)}
\arguments{
  \item{y}{The vector of observations to segment.}
  \item{base}{Position of each observation in base pairs.}
  \item{starts}{Starts of the 1-annotated regions in base pairs.}
  \item{ends}{Ends of the 1-annotated regions in base pairs.}
}

\value{List of results describing the segmentation model. In addition to
the results of the SegAnnot function, there are data.frames
seg.df, break.df, signal.df, and ann.df which can be easily
plotted for model interpretation.}

\author{Guillem Rigaill, Toby Dylan Hocking}




\examples{
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
}
