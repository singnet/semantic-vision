from atomspace cimport get_value_ptr

cdef Value wrapPtrValue(cValuePtr shared_ptr):
    """Factory method to construct PtrValue from C++ ValuePtr (see
    http://docs.cython.org/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers
    for example)"""
    cdef PtrValue value = PtrValue.__new__(PtrValue)
    value.shared_ptr = shared_ptr
    return value

cdef class PtrValue(Value):
    def value(self):
        return <object>(<cPtrValue*>get_value_ptr(self)).value()
