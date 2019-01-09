OPENCOG_ADD_ATOM_TYPES(tensor_atom_types.script
	tensor_atom_types.h
	tensor_atom_types.definitions
	tensor_atom_types.inheritance
	tensor_atom_types.scm
	tensor_atom_types.pyx)

ADD_CUSTOM_TARGET(tensor_atom_types DEPENDS tensor_atom_types.h
	tensor_atom_types.definitions tensor_atom_types.inheritance
	tensor_atom_types.scm tensor_atom_types.pyx)

# The atom_types.h file is written to the build directory
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

ADD_LIBRARY(tensor_types SHARED
	tensor_atom_types_init.cc
)

ADD_DEPENDENCIES(tensor_types tensor_atom_types)
