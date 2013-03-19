works_with_R("2.15.2", SegAnnot="1.1")

data(profiles)

pro <- profiles$hi
for(N in names(pro)){
  f <- sprintf("%s.csv", N)
  df <- pro[[N]]
  write.csv(df, f, row.names=FALSE,quote=FALSE)
}

ann <- subset(pro$ann, annotation=="1breakpoint")
result <- SegAnnotBases(pro$pro$log, pro$pro$pos, ann$min, ann$max)
write.csv(result$seg.df, "segmentation-R.csv", row.names=FALSE, quote=FALSE)
