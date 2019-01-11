from opencog.atomspace cimport get_value_ptr

cdef Value wrapPtrValue(cValuePtr shared_ptr):
    """Factory method to construct PtrValue from C++ ValuePtr (see
    http://docs.cython.org/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers
    for example)"""
    cdef PtrValueClass value = PtrValueClass.__new__(PtrValueClass)
    value.shared_ptr = shared_ptr
    return value

cdef class PtrValueClass(Value):
    def value(self):
        return <object>(<cPtrValue*>get_value_ptr(self)).value()

def PtrValue(obj):
    incref(<void*>obj)
    return wrapPtrValue(createPtrValue(<void*>obj, decref))
