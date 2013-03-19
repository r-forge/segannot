#include <Python.h>
#include <numpy/arrayobject.h>
#include "SegAnnot.h"

static PyObject *
SegAnnotBases2Py(PyObject *self, PyObject *args){
    PyArrayObject *signal, *base, *starts, *ends; //borrowed
    if(!PyArg_ParseTuple(args, "O!O!O!O!",
			 &PyArray_Type, &signal,
			 &PyArray_Type, &base,
			 &PyArray_Type, &starts,
			 &PyArray_Type, &ends
	   )){
	return NULL;
    }
    if(PyArray_TYPE(signal)!=PyArray_DOUBLE){
	PyErr_SetString(PyExc_TypeError,
			"signal must be numpy.ndarray type double");
	return NULL;
    }
    if(PyArray_TYPE(base)!=PyArray_INT){
	PyErr_SetString(PyExc_TypeError,
			"base must be numpy.ndarray type int");
	return NULL;
    }
    if(PyArray_TYPE(starts)!=PyArray_INT){
	PyErr_SetString(PyExc_TypeError,
			"starts must be numpy.ndarray type int");
	return NULL;
    }
    if(PyArray_TYPE(ends)!=PyArray_INT){
	PyErr_SetString(PyExc_TypeError,
			"ends must be numpy.ndarray type int");
	return NULL;
    }
    npy_intp n_signal = PyArray_DIM(signal,0);
    npy_intp n_base = PyArray_DIM(base,0);
    if(n_signal != n_base){
	PyErr_SetString(PyExc_TypeError,
			"signal and base must be same length");
	return NULL;
    }
    npy_intp n_starts = PyArray_DIM(starts,0);
    npy_intp n_ends = PyArray_DIM(ends,0);
    if(n_starts != n_ends){
	PyErr_SetString(PyExc_TypeError,
			"starts and ends must be same length");
	return NULL;
    }
    double *signalA = (double*)PyArray_DATA(signal);
    int *baseA = (int*)PyArray_DATA(base);
    int *startsA = (int*)PyArray_DATA(starts);
    int *endsA = (int*)PyArray_DATA(ends);
    // Initialize data for return vals.
    npy_intp n_segments = n_starts+1;
    PyObject *segStart = PyArray_SimpleNew(1,&n_segments,PyArray_INT);
    int *segStartA = (int*)PyArray_DATA(segStart);
    PyObject *segEnd = PyArray_SimpleNew(1,&n_segments,PyArray_INT);
    int *segEndA = (int*)PyArray_DATA(segEnd);
    PyObject *bkpts = PyArray_SimpleNew(1,&n_starts,PyArray_INT);
    int *bkptsA = (int*)PyArray_DATA(bkpts);
    PyObject *segMean = PyArray_SimpleNew(1,&n_segments,PyArray_DOUBLE);
    double *segMeanA = (double*)PyArray_DATA(segMean);
    int status = SegAnnotBases( 
	signalA, baseA, startsA, endsA, 
	n_signal, n_starts, 
	segStartA, segEndA, bkptsA, segMeanA);
    if(status == ERROR_BASES_NOT_INCREASING){
	PyErr_SetString(PyExc_TypeError,
			"bases not increasing");
    }
    if(status == ERROR_REGIONS_NOT_INCREASING){
	PyErr_SetString(PyExc_TypeError,
			"regions not increasing");
    }
    if(status == ERROR_LAST_BEFORE_FIRST){
	PyErr_SetString(PyExc_TypeError,
			"last base of region before first");
    }
    if(status != 0){
	return NULL;
    }

    return Py_BuildValue("{s:N,s:N,s:N,s:N}",
			 "start",segStart,
			 "end",segEnd,
			 "bkpts",bkpts,
			 "mean",segMean);
}

static PyMethodDef Methods[] = {
  {"SegAnnotBases", SegAnnotBases2Py, METH_VARARGS, 
   "Find the L2-optimal segmentation for complete 1-annotated regions"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initSegAnnot
(void){
    (void)Py_InitModule("SegAnnot",Methods);
    import_array();//necessary from numpy otherwise we crash with segfault
}
