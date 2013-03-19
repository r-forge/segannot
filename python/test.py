from SegAnnot import SegAnnotBases
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
            if name == "annotation" and item == "0breakpoints":
                pass
            else:
                data[prefix][name].append(fun(item))

arrays = {}
for prefix, d in data.iteritems():
    for name, items in d.iteritems():
        arrays[prefix+"_"+name] = numpy.array(items)

result = SegAnnotBases(arrays["probes_logratio"],
                       arrays["probes_position"],
                       arrays["annotations_min"],
                       arrays["annotations_max"])

#out = open("segmentation-python.csv", "w")
