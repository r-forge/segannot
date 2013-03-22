works_with_R("2.15.2", SegAnnot="1.2", cghseg="1.0.1")

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

## quick check that the path is correct.
cghseg:::colibriR_c(pro$pro[1:10,"logratio"],4)$path

endmat <- cghseg:::segmeanCO(pro$pro$log,5)$t.est
write.table(endmat,"pruned-dp-R.csv",quote=FALSE,
            row.names=FALSE,col.names=FALSE)
