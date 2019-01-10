OPENCOG_ADD_ATOM_TYPES(neuralnet_atom_types.script
	neuralnet_atom_types.h
	neuralnet_atom_types.definitions
	neuralnet_atom_types.inheritance
	neuralnet_atom_types.scm
	neuralnet_atom_types.pyx)

ADD_CUSTOM_TARGET(neuralnet_atom_types DEPENDS neuralnet_atom_types.h
	neuralnet_atom_types.definitions neuralnet_atom_types.inheritance
	neuralnet_atom_types.scm neuralnet_atom_types.pyx)

ADD_LIBRARY(opencog-neuralnet SHARED
	neuralnet_atom_types_init.cc
	PtrValue.cc
)

TARGET_LINK_LIBRARIES(opencog-neuralnet
	${ATOMSPACE_atomcore_LIBRARY}
)

ADD_DEPENDENCIES(opencog-neuralnet neuralnet_atom_types)

INSTALL (FILES
	PtrValue.h
	DESTINATION "include/opencog/neuralnet"
)
