LINK_LIBRARIES(
	opencog-neuralnet
	${ATOMSPACE_atomcore_LIBRARY}
)

ADD_CXXTEST(PtrValueUTest)
