#include <Python.h>
#include <numpy/arrayobject.h>
#include "PrunedDP.h"

static PyObject *
PrunedDP2Py(PyObject *self, PyObject *args){
    PyArrayObject *signal;
    int Kmax;
    if(!PyArg_ParseTuple(args, "O!i",
			 &PyArray_Type, &signal,
			 &Kmax)){
	return NULL;
    }
    if(PyArray_TYPE(signal)!=PyArray_DOUBLE){
	PyErr_SetString(PyExc_TypeError,
			"signal must be numpy.ndarray type double");
	return NULL;
    } 
    npy_intp n_signal = PyArray_DIM(signal, 0);
    npy_intp n_path[] = {Kmax, n_signal};
    PyObject *path = PyArray_SimpleNew(2,n_path,PyArray_INT);
    int *pathA = (int*)PyArray_DATA(path);
    double *signalA = (double*)PyArray_DATA(signal);
    int status = PrunedDP(signalA, n_signal, Kmax, pathA);
    if(status != 0){
	PyErr_SetString(PyExc_TypeError,
			"signal/kmax too small");
	return NULL;
    }
    return path;
}

static PyMethodDef Methods[] = {
  {"PrunedDP", PrunedDP2Py, METH_VARARGS,
   "L2-optimal segmentation from 1 to Kmax segments"},
  {NULL, NULL, 0, NULL}
};

extern "C"
PyMODINIT_FUNC
initPrunedDP
(void){
    (void)Py_InitModule("PrunedDP",Methods);
    import_array();//necessary from numpy otherwise we crash with segfault
}
