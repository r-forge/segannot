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
  The format is:
List of 2
 $ low.resolution :List of 2
  ..$ probes     :'data.frame':	123 obs. of  2 variables:
  .. ..$ position: int [1:123] 396626 543078 761031 814919 869470 1925824 3842683 5841351 6840798 7480253 ...
  .. ..$ logratio: num [1:123] 0.0384 -0.0589 -0.016 0.0215 -0.0954 ...
  ..$ annotations:'data.frame':	3 obs. of  3 variables:
  .. ..$ min       : int [1:3] 18550922 56833281 58857018
  .. ..$ max       : int [1:3] 41374179 58575943 60543465
  .. ..$ annotation: Factor w/ 2 levels "0breakpoints",..: 2 2 2
 $ high.resolution:List of 2
  ..$ probes     :'data.frame':	153663 obs. of  2 variables:
  .. ..$ position: int [1:153663] 2785 5703 6412 8674 11833 12493 14077 14819 19443 19631 ...
  .. ..$ logratio: num [1:153663] 0.3141 -0.0647 0.152 0.2596 0.5911 ...
  ..$ annotations:'data.frame':	8 obs. of  3 variables:
  .. ..$ min       : int [1:8] 14942223 16002142 35785637 106185798 106247811 6790040 5117834 35838384
  .. ..$ max       : int [1:8] 15930150 16958707 35835533 106247811 106299845 7053772 5524837 36032263
  .. ..$ annotation: Factor w/ 2 levels "0breakpoints",..: 2 2 2 2 2 1 1 2
}
\examples{
data(profiles,package="SegAnnot")
pro <- profiles$low
fit <- SegAnnotBases(pro$pro$log,pro$pro$pos,pro$ann$min,pro$ann$max)
library(ggplot2)
p <- ggplot()+
  geom_tallrect(aes(xmin=min/1e6,xmax=max/1e6,fill=annotation),
                data=pro$ann)+
  geom_point(aes(position/1e6,logratio),data=pro$pro)+
  scale_fill_manual(values=breakpoint.colors)
print(p)
}
\keyword{datasets}
