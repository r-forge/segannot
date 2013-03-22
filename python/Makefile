LIBDIR=build/lib.linux-x86_64-2.7

compare.pdf: compare.R start.txt
	R --no-save < $<
start.txt: ${LIBDIR}/SegAnnot.so ${LIBDIR}/PrunedDP.so test.py probes.csv
	PYTHONPATH=${LIBDIR} time python test.py
${LIBDIR}/SegAnnot.so, ${LIBDIR}/PrunedDP.so: SegAnnot.c SegAnnot_interface.c SegAnnot.h setup.py PrunedDP.cpp PrunedDP_interface.cpp PrunedDP.h polynome.h 
	python setup.py build
probes.csv: profile.R
	R --no-save < $<