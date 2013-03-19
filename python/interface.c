/* -*- compile-command: "python setup.py build"; compilation-read-command: nil -*- */ 
#include <Python.h>
#include <numpy/arrayobject.h>
#include "SegAnnot.h"

static PyObject *
SegAnnotBases2Py(PyObject *self, PyObject *args){
  PyArrayObject *data; //borrowed
  if(!PyArg_ParseTuple(args,"O!",&PyArray_Type,&data))return NULL;
  if(PyArray_TYPE(data)!=PyArray_DOUBLE){
    PyErr_SetString(PyExc_TypeError,"Input must be numpy.ndarray type double");
    return NULL;
  }
  double *x = (double*)PyArray_DATA(data);
  int n = PyArray_DIM(data,0);
  PyObject *result;
  return result;
}

static PyMethodDef Methods[] = {
  {"SegAnnotBases", SegAnnotBases2Py, METH_VARARGS, 
   "Find the L2-optimal segmentation for complete 1-annotated regions"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initSegAnnot
(void){
    import_array();//necessary from numpy otherwise we crash with segfault
    PyObject *m=Py_InitModule("SegAnnot",Methods);
    if(m==NULL)
	return;
}
