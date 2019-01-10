#include <Python.h>

void incref(void* ptr)
{
	Py_INCREF(static_cast<PyObject*>(ptr));
}

void decref(void* ptr)
{
	Py_DECREF(static_cast<PyObject*>(ptr));
}
