LIBDIR=build/lib.linux-x86_64-2.7

compare.pdf: compare.R start.txt
	R --no-save < $<
start.txt: ${LIBDIR}/SegAnnot.so test.py probes.csv
	PYTHONPATH=${LIBDIR} time python test.py
${LIBDIR}/SegAnnot.so: SegAnnot.c interface.c SegAnnot.h setup.py
	python setup.py build
probes.csv: profile.R
	R --no-save < $<