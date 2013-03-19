#include <Python.h>
#include <numpy/arrayobject.h>
#include "SegAnnot.h"

/* PyObject * tree2py(Cluster *c){ */
/*   PyObject *alpha,*lambda,*i; */
/*   double *alpha_ptr,*lambda_ptr; */
/*   int *i_ptr; */
/*   npy_intp dim=c->total; */
/*   alpha =PyArray_SimpleNew(1,&dim,PyArray_DOUBLE); */
/*   lambda=PyArray_SimpleNew(1,&dim,PyArray_DOUBLE); */
/*   i     =PyArray_SimpleNew(1,&dim,PyArray_INT); */
/*   alpha_ptr =(double*)PyArray_DATA(alpha); */
/*   lambda_ptr=(double*)PyArray_DATA(lambda); */
/*   i_ptr     =(int*)   PyArray_DATA(i); */
/*   int row=0; */
/*   add_results(c,alpha_ptr,lambda_ptr,i_ptr,&row); */
/*   return Py_BuildValue("{s:O,s:O,s:O}", */
/* 		       "alpha",alpha, */
/* 		       "lambda",lambda, */
/* 		       "i",i); */
/* } */

static PyObject *
SegAnnotBases2Py(PyObject *self, PyObject *args){
    PyArrayObject *signal, *base, *starts, *ends, *segStart; //borrowed
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
    if(PyArray_TYPE(base)!=PyArray_INT64){
	PyErr_SetString(PyExc_TypeError,
			"base must be numpy.ndarray type int");
	return NULL;
    }
    if(PyArray_TYPE(starts)!=PyArray_INT64){
	PyErr_SetString(PyExc_TypeError,
			"starts must be numpy.ndarray type int");
	return NULL;
    }
    if(PyArray_TYPE(ends)!=PyArray_INT64){
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
    segStart = PyArray_SimpleNew(1,&n_starts,PyArray_INT);
    int *segStartA = (int*)PyArray_DATA(segStart);
    int status = SegAnnotBases(
	signalA, baseA, startsA, endsA, 
	n_signal, n_starts, segStartA);
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
    return segStart;
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
