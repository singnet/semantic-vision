# TODO: replace by include from atomspace.pxd
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

ctypedef short Type

cdef extern from "opencog/atoms/value/Value.h" namespace "opencog":
    cdef cppclass cValue "opencog::Value":
        Type get_type()
        bint is_atom()
        bint is_node()
        bint is_link()

        string to_string()
        string to_short_string()
        bint operator==(const cValue&)
        bint operator!=(const cValue&)

    ctypedef shared_ptr[cValue] cValuePtr "opencog::ValuePtr"

cdef class Value:
    cdef cValuePtr shared_ptr

cdef cValue* get_value_ptr(Value protoAtom)
