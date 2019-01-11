INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

# TODO: replace by cogutil include
SET(CYTHON_FLAGS "-f")
# Cythonizes one .pyx file into a .cpp file
# Additional arguments are dependencies
MACRO(CYTHON_ADD_MODULE_PYX name)
	# Allow the user to specify dependencies as optional arguments
	SET(DEPENDS ${DEPENDS} ${ARGN})
	ADD_CUSTOM_COMMAND(
		OUTPUT ${name}.cc
		COMMAND ${CYTHON_EXECUTABLE}
		ARGS ${CYTHON_FLAGS}
				-I ${PROJECT_BINARY_DIR}
				-I ${CMAKE_CURRENT_SOURCE_DIR}
				-o ${name}.cc
				--cplus ${CMAKE_CURRENT_SOURCE_DIR}/${name}
		DEPENDS ${ARGN}
		COMMENT "Cythonizing ${name} to ${name}.cc")
ENDMACRO(CYTHON_ADD_MODULE_PYX)

ADD_SUBDIRECTORY(neuralnet)
