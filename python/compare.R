works_with_R("2.15.2", ggplot2="0.9.3")

dp.R <- scan("pruned-dp-R.csv")
dp.python <- scan("pruned-dp-python.csv")
dp.R[dp.R!=0] <- dp.R[dp.R!=0]-1
stopifnot(all(dp.python == dp.R))

seg.R <- read.csv("segmentation-R.csv")

seg.py <- list()
for(pre in c("mean","start","end")){
  f <- sprintf("%s.txt", pre)
  seg.py[[pre]] <- scan(f)
}
seg.py <- do.call(data.frame,seg.py)

stopifnot(abs(seg.py$mean - seg.R$mean) < 1e-6)
stopifnot(all(abs(seg.py$start - seg.R$first.base)<=0.5))
stopifnot(all(abs(seg.py$end - seg.R$last.base)<=0.5))
seg.R <- transform(seg.R,start=first.base,end=last.base)

df <- rbind(data.frame(seg.py,interface="python"),
            data.frame(seg.R[,colnames(seg.py)],interface="R"))

p <- ggplot(df,aes(start,mean,colour=interface))+
  geom_segment(aes(xend=end,yend=mean,size=interface))+
  geom_point()+
  scale_size_manual(values=c(R=1,python=2))

pdf("compare.pdf")
print(p)
dev.off()
