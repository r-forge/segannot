from SegAnnot import SegAnnotBases
from PrunedDP import PrunedDP
import csv
import numpy

TABLES = {
    "annotations":(int,int,str),
    "probes":(int,float),
}

data = {}
for prefix, types in TABLES.iteritems():
    fn = prefix+".csv"
    f = open(fn)
    r = csv.reader(f)
    header = r.next()
    data[prefix] = {}
    for name in header:
        data[prefix][name] = []
    for tup in r:
        entries = zip(header, tup, types)
        for name, item, fun in entries:
            data[prefix][name].append(fun(item))
## re-order annotations.
ann_tups = zip(*data["annotations"].values())
ann_tups.sort()
for i, k in enumerate(data["annotations"]):
    data["annotations"][k] = [t[i] for t in ann_tups if t[1]=="1breakpoint"]
dtypes={
    "min":numpy.int32,
    "max":numpy.int32,
    "position":numpy.int32,
    "logratio":numpy.float,
    "annotation":numpy.str,
}
arrays = {}
for prefix, d in data.iteritems():
    for name, items in d.iteritems():
        arrays[prefix+"_"+name] = numpy.array(items,dtypes[name])
result = SegAnnotBases(arrays["probes_logratio"],
                       arrays["probes_position"],
                       arrays["annotations_min"],
                       arrays["annotations_max"])
print result
for name, a in result.iteritems():
    out = name + ".txt"
    a.tofile(out," ")
 
short = arrays["probes_logratio"]
result = PrunedDP(short, 5)
result.tofile("pruned-dp-python.csv",sep=" ")

