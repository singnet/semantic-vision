CYTHON_ADD_MODULE_PYX(ptrvalue.pyx "ptrvalue.pxd" "ptrvalue.h")

ADD_LIBRARY(opencog-neuralnet-cython SHARED
	ptrvalue.pyx.cc
	ptrvalue.cc
)

TARGET_LINK_LIBRARIES(opencog-neuralnet-cython
	opencog-neuralnet
)
