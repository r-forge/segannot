from SegAnnot import SegAnnotBases
import numpy
d =SegAnnotBases(
    numpy.array([2.0, 3.0, -2.0, -1.0, 5.0, 6.0]),
    numpy.array([5, 6, 7, 8, 9, 10], numpy.int32),
    numpy.array([5, 8], numpy.int32),
    numpy.array([7, 10], numpy.int32))
print d
print d["end"] == numpy.array([6, 8, 10])
print d["start"] == numpy.array([5, 7, 9])
print d["break_mid"] == numpy.array([6.5, 8.5])
