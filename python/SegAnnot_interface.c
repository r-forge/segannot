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
    PyArray_Descr *signal_dtype = PyArray_DTYPE(signal);
    if(!PyArray_CanCastSafely(signal_dtype->type_num, NPY_DOUBLE)){
	PyErr_Format
	  (PyExc_TypeError,
	   "signal[%c%d] must be safely castable to double",
	   signal_dtype->type, signal_dtype->elsize);
	return NULL;
    }
    PyArray_Descr *base_dtype = PyArray_DTYPE(base);
    if(!PyArray_CanCastSafely(base_dtype->type_num, NPY_INT)){
	PyErr_Format
	  (PyExc_TypeError,
	   "base[%c%d] must be safely castable to int",
	   base_dtype->type, base_dtype->elsize);
	return NULL;
    }
    PyArray_Descr *starts_dtype = PyArray_DTYPE(starts);
    if(!PyArray_CanCastSafely(starts_dtype->type_num, NPY_INT)){
	PyErr_Format
	  (PyExc_TypeError,
	   "starts[%c%d] must be safely castable to int",
	   starts_dtype->type, starts_dtype->elsize);
	return NULL;
    }
    PyArray_Descr *ends_dtype = PyArray_DTYPE(ends);
    if(!PyArray_CanCastSafely(ends_dtype->type_num, NPY_INT)){
	PyErr_Format
	  (PyExc_TypeError,
	   "ends[%c%d] must be safely castable to int",
	   ends_dtype->type, ends_dtype->elsize);
	return NULL;
    }
    npy_intp n_signal = PyArray_DIM(signal,0);
    npy_intp n_base = PyArray_DIM(base,0);
    if(n_signal != n_base){
	PyErr_SetString(PyExc_ValueError,
			"signal and base must be same length");
	return NULL;
    }
    npy_intp n_starts = PyArray_DIM(starts,0);
    npy_intp n_ends = PyArray_DIM(ends,0);
    if(n_starts != n_ends){
	PyErr_SetString(PyExc_ValueError,
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
    PyObject *break_min = PyArray_SimpleNew(1,&n_starts,PyArray_INT);
    int *break_minA = (int*)PyArray_DATA(break_min);
    PyObject *break_mid = PyArray_SimpleNew(1,&n_starts,PyArray_DOUBLE);
    double *break_midA = (double*)PyArray_DATA(break_mid);
    PyObject *break_max = PyArray_SimpleNew(1,&n_starts,PyArray_INT);
    int *break_maxA = (int*)PyArray_DATA(break_max);
    PyObject *segMean = PyArray_SimpleNew(1,&n_segments,PyArray_DOUBLE);
    double *segMeanA = (double*)PyArray_DATA(segMean);
    int status = SegAnnotBases( 
	signalA, baseA, startsA, endsA, 
	n_signal, n_starts, 
	segStartA, segEndA, segMeanA,
	break_minA, break_midA, break_maxA);
    if(status == ERROR_BASES_NOT_INCREASING){
	PyErr_SetString(PyExc_ValueError,
			"bases not increasing");
    }
    if(status == ERROR_REGIONS_NOT_INCREASING){
	PyErr_SetString(PyExc_ValueError,
			"regions not increasing");
    }
    if(status == ERROR_LAST_BEFORE_FIRST){
	PyErr_SetString(PyExc_ValueError,
			"last base of region before first");
    }
    if(status != 0){
	return NULL;
    }

    return Py_BuildValue("{s:N,s:N,s:N,s:N,s:N,s:N}",
			 "start",segStart,
			 "end",segEnd,
			 "mean",segMean,
			 "break_min",break_min,
			 "break_mid",break_mid,
			 "break_max",break_max);
}

static PyMethodDef Methods[] = {
  {"SegAnnotBases", SegAnnotBases2Py, METH_VARARGS, 
   "L2-optimal segmentation for complete 1-annotated regions"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initSegAnnot
(void){
    (void)Py_InitModule("SegAnnot",Methods);
    import_array();//necessary from numpy otherwise we crash with segfault
}
