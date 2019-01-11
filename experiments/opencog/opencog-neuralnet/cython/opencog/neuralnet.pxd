from libcpp.memory cimport shared_ptr
from opencog.atomspace cimport Type, Value, cValue, cValuePtr

cdef extern from "cython/opencog/ptrvalue.h":
    cdef void incref(void* ptr)
    cdef void decref(void* ptr)

cdef Value wrapPtrValue(cValuePtr shared_ptr)

cdef extern from "opencog/neuralnet/PtrValue.h" namespace "opencog":
    cdef cppclass cPtrValue "opencog::PtrValue":
        void* value() const;

    cdef cValuePtr createPtrValue(...)
