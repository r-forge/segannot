\name{profiles}
\alias{profiles}
\docType{data}
\title{
Copy number profiles and complete 0/1 breakpoint annotations
}
\description{
This data set contains two motivating segmentation problems for the
SegAnnot algorithm. As described in the vignette,
maximum-likelihood segmentations can be calculated using
cghseg::segmeanCO, but none of the segmentations are consistent with the
given breakpoint annotations. So SegAnnot can be used in these cases to
find the best consistent segmentation in terms of the square loss.
}
\usage{data(profiles)}
\format{
  A list of two separate segmentation problems, named low.resolution and
  high.resolution. Each problem is a list of two data.frames:
  probes and annotations. The probes come from a microarray experiment,
  and the
  annotations come from an expert's interpretation of a scatterplot of
  the probe logratio versus position.
}
\examples{
data(profiles,package="SegAnnot")
pro <- profiles$low
library(ggplot2)
p <- ggplot()+
  geom_tallrect(aes(xmin=min/1e6,xmax=max/1e6,fill=annotation),
                data=pro$ann)+
  geom_point(aes(position/1e6,logratio),data=pro$pro)+
  scale_fill_manual(values=breakpoint.colors)
print(p)
fit <- SegAnnotBases(pro$pro$log,pro$pro$pos,pro$ann$min,pro$ann$max)
pfit <- p+
  geom_segment(aes(first.base/1e6,mean,xend=last.base/1e6,yend=mean),
    data=fit$seg.df,colour=signal.colors[["estimate"]],lwd=2)+
  geom_vline(aes(xintercept=base/1e6),data=fit$break.df,
    colour=signal.colors[["estimate"]],lwd=2,linetype="dashed")
print(pfit)
}
\keyword{datasets}
