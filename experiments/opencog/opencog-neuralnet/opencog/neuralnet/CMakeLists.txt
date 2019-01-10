OPENCOG_ADD_ATOM_TYPES(neuralnet_atom_types.script
	neuralnet_atom_types.h
	neuralnet_atom_types.definitions
	neuralnet_atom_types.inheritance
	neuralnet_atom_types.scm
	neuralnet_atom_types.pyx)

ADD_CUSTOM_TARGET(neuralnet_atom_types DEPENDS neuralnet_atom_types.h
	neuralnet_atom_types.definitions neuralnet_atom_types.inheritance
	neuralnet_atom_types.scm neuralnet_atom_types.pyx)

# The atom_types.h file is written to the build directory
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

ADD_LIBRARY(opencog-neuralnet SHARED
	neuralnet_atom_types_init.cc
)

ADD_DEPENDENCIES(opencog-neuralnet neuralnet_atom_types)

INSTALL (FILES
	DESTINATION "include/opencog/neuralnet"
)
