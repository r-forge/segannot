LIBDIR=build/lib.linux-x86_64-2.7

segmentation-python.csv: ${LIBDIR}/SegAnnot.so test.py probes.csv
	PYTHONPATH=${LIBDIR} time python test.py
${LIBDIR}/SegAnnot.so: SegAnnot.c interface.c SegAnnot.h setup.py
	python setup.py build
probes.csv: profile.R
	R --no-save < $<