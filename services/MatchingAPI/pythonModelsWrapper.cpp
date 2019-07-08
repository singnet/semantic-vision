#include <iostream>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "string"
#include <memory>
#define NPY_NO_DEPRECATED_APINPY_1_7_API_VERSION

using namespace std;

class PyObjectPtr {
    PyObject* obj;
public:
    PyObjectPtr(PyObject* obj) : obj(obj) { }
    ~PyObjectPtr() {
        Py_DECREF(obj);
    }

    PyObject* get() {
        return obj;
    }

};

void call_getMagicPointKps(string image, double threshold)
{
    const char* pyName = "getSuperPointKPs";

    PyObject* sysPath = PySys_GetObject("path");
    PyObjectPtr upperDirStr = PyUnicode_FromString("..");
    PyList_Append(sysPath, upperDirStr.get());
    PyErr_Print();

    PyObjectPtr pName = PyUnicode_FromString(pyName);
    PyObject* pModule = PyImport_ImportModule(pyName);
    PyObject* pFunc = PyObject_GetAttrString(pModule, "getMagicPointKps");
    PyObjectPtr pArgs(PyTuple_New(2));
    PyTuple_SetItem(pArgs.get(), 0, PyBytes_FromStringAndSize(image.c_str(), Py_ssize_t(image.size())));
    PyTuple_SetItem(pArgs.get(), 1, PyFloat_FromDouble(threshold));
    PyObjectPtr pValue(PyObject_CallObject(pFunc, pArgs.get()));
    PyObject* xs = PyTuple_GetItem(pValue.get(), 0);
    PyObject* ys = PyTuple_GetItem(pValue.get(), 1);
    long* a_x = (long *)PyArray_DATA(xs);
    long* a_y = (long *)PyArray_DATA(ys);
}

void call_getSuperPointKps(string image, double threshold)
{
    const char* pyName = "getSuperPointKPs";

    PyObject* sysPath = PySys_GetObject("path");
    PyObjectPtr upperDirStr = PyUnicode_FromString("..");
    PyList_Append(sysPath, upperDirStr.get());
    PyErr_Print();

    PyObjectPtr pName = PyUnicode_FromString(pyName);
    PyObject* pModule = PyImport_ImportModule(pyName);
    PyObject* pFunc = PyObject_GetAttrString(pModule, "getSuperPointKps");
    PyObjectPtr pArgs(PyTuple_New(2));
    PyTuple_SetItem(pArgs.get(), 0, PyBytes_FromStringAndSize(image.c_str(), Py_ssize_t(image.size())));
    PyTuple_SetItem(pArgs.get(), 1, PyFloat_FromDouble(threshold));
    PyObjectPtr pValue(PyObject_CallObject(pFunc, pArgs.get()));
    PyObject* xs = PyTuple_GetItem(pValue.get(), 0);
    PyObject* ys = PyTuple_GetItem(pValue.get(), 1);
    long* a_x = (long *)PyArray_DATA(xs);
    long* a_y = (long *)PyArray_DATA(ys);
}

void call_getSuperPointDescriptors(string image, double threshold)
{
    const char* pyName = "getSuperPointKPs";

    PyObject* sysPath = PySys_GetObject("path");
    PyObjectPtr upperDirStr = PyUnicode_FromString("..");
    PyList_Append(sysPath, upperDirStr.get());
    PyErr_Print();

    PyObjectPtr pName = PyUnicode_FromString(pyName);
    PyObject* pModule = PyImport_ImportModule(pyName);
    PyObject* pFunc = PyObject_GetAttrString(pModule, "getSuperPointDescriptors");
    PyObjectPtr pArgs(PyTuple_New(2));
    PyTuple_SetItem(pArgs.get(), 0, PyBytes_FromStringAndSize(image.c_str(), Py_ssize_t(image.size())));
    PyTuple_SetItem(pArgs.get(), 1, PyFloat_FromDouble(threshold));
    PyObjectPtr pValue(PyObject_CallObject(pFunc, pArgs.get()));
    PyObject* xs = PyTuple_GetItem(PyTuple_GetItem(pValue.get(), 0), 0);
    PyObject* ys = PyTuple_GetItem(PyTuple_GetItem(pValue.get(), 0), 1);
    PyObject* desc = PyTuple_GetItem(pValue.get(), 1);
    long* a_x = (long *)PyArray_DATA(xs);
    long* a_y = (long *)PyArray_DATA(ys);
    float* descs = (float*)PyArray_DATA(desc);
    int height = PyArray_DIM(desc, 0);
    int width = PyArray_DIM(desc, 1);
}

static string getImageString(string path)
{
    FILE *in_file  = fopen(path.c_str(), "rb");

    fseek(in_file, 0L, SEEK_END);
    int sz = ftell(in_file);
    rewind(in_file);
    char imageBytes[sz];
    fread(imageBytes, sizeof *imageBytes, sz, in_file);
    string image_bytes(imageBytes, sz);
    return image_bytes;
}

//int main(int argc, char *argv[])
//{
//    string image("../Woods.jpg");
//    string image_bytes = getImageString(image);
//    Py_Initialize();
//    import_array();
//    call_getMagicPointKps(image_bytes, 0.015);
//    call_getSuperPointDescriptors(image_bytes, 0.015);
//    Py_Finalize();
//}
