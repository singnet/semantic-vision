CYTHON_ADD_MODULE_PYX(neuralnet.pyx
	"neuralnet.pxd" "ptrvalue.pyx" "ptrvalue.h" "ptrvalue.cc"
	"atomspace.pxd")

ADD_LIBRARY(cython-opencog-neuralnet SHARED
	neuralnet.pyx.cc
	ptrvalue.cc
)

TARGET_LINK_LIBRARIES(cython-opencog-neuralnet
	opencog-neuralnet
)

# override name of the .so file to align with Python standards
SET_TARGET_PROPERTIES(cython-opencog-neuralnet PROPERTIES
	PREFIX ""
	OUTPUT_NAME neuralnet)
