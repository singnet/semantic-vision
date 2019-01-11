CYTHON_ADD_MODULE_PYX(ptrvalue.pyx "ptrvalue.pxd" "ptrvalue.h")

ADD_LIBRARY(cython-neuralnet-ptrvalue SHARED
	ptrvalue.pyx.cc
	ptrvalue.cc
)

TARGET_LINK_LIBRARIES(cython-neuralnet-ptrvalue
	opencog-neuralnet
)

SET_TARGET_PROPERTIES(cython-neuralnet-ptrvalue PROPERTIES
	PREFIX ""
	OUTPUT_NAME ptrvalue)
