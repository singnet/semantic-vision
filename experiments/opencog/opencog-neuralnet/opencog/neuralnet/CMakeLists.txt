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

SET_TARGET_PROPERTIES(opencog-neuralnet PROPERTIES VERSION ${PROJECT_VERSION})

INSTALL(TARGETS opencog-neuralnet
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/opencog
)

INSTALL(FILES PtrValue.h ${CMAKE_CURRENT_BINARY_DIR}/neuralnet_atom_types.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opencog/neuralnet
)

