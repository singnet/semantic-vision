CYTHON_ADD_MODULE_PYX(neuralnet.pyx
	"neuralnet.pxd" "ptrvalue.pyx" "ptrvalue.h" "ptrvalue.cc")

ADD_LIBRARY(cython-opencog-neuralnet SHARED
	neuralnet.pyx.cc
	ptrvalue.cc
)

TARGET_LINK_LIBRARIES(cython-opencog-neuralnet
	opencog-neuralnet
)

# override name of the .so file to align with Python standards
SET_TARGET_PROPERTIES(cython-opencog-neuralnet PROPERTIES
	VERSION ${PROJECT_VERSION}
	PREFIX ""
	OUTPUT_NAME neuralnet)

INSTALL(TARGETS cython-opencog-neuralnet
	DESTINATION ${PYTHON_DEST}/opencog
)

# don't install __init__.py because it should already be installed by atomspace
INSTALL (FILES neuralnet.pxd
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opencog/cython/opencog
)
