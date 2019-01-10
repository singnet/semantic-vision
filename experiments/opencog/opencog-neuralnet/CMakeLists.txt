# ===================================================================
# Build system settings

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.2)
IF(CMAKE_VERSION VERSION_GREATER 3.0.2)
	CMAKE_POLICY(SET CMP0037 OLD)
ENDIF(CMAKE_VERSION VERSION_GREATER 3.0.2)

PROJECT(opencog-neuralnet)

# set default build type
IF (CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Release)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "")

# DATADIR is required by OpenCogMacros.cmake
IF (NOT DEFINED DATADIR)
	SET (DATADIR "${CMAKE_INSTALL_PREFIX}/share/opencog")
ENDIF (NOT DEFINED DATADIR)
ADD_DEFINITIONS(-DDATADIR="${DATADIR}")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
# The neuralnet_atom_types.h file is written to the build directory
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

# TODO: replace by include from cogutil if PR
# https://github.com/opencog/cogutil/pull/159 is merged
IF (CMAKE_COMPILER_IS_GNUCXX)
	# Atomspace is already asking for version 5.0, so ask for that.
	IF (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
		MESSAGE(FATAL_ERROR "GCC version must be at least 5.0!")
	ENDIF (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)

	SET(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=gnu++11")
ENDIF (CMAKE_COMPILER_IS_GNUCXX)

# ===================================================================
# Dependencies

# CogUtil
FIND_PACKAGE(CogUtil 2.0.3 CONFIG REQUIRED)
IF (COGUTIL_FOUND)
    MESSAGE(STATUS "CogUtil found.")
ELSE (COGUTIL_FOUND)
    MESSAGE(FATAL_ERROR "CogUtil missing: it is needed!")
ENDIF (COGUTIL_FOUND)

LIST(APPEND CMAKE_MODULE_PATH  ${COGUTIL_DATA_DIR}/cmake)
INCLUDE(AddCxxtest)
INCLUDE(Summary)

# AtomSpace
FIND_PACKAGE(AtomSpace 5.0.4 CONFIG REQUIRED)
IF (ATOMSPACE_FOUND)
	MESSAGE(STATUS "AtomSpace found.")
ELSE (ATOMSPACE_FOUND)
	MESSAGE(FATAL_ERROR "AtomSpace missing: it is needed!")
ENDIF (ATOMSPACE_FOUND)

LIST(APPEND CMAKE_MODULE_PATH  ${ATOMSPACE_DATA_DIR}/cmake)
INCLUDE("OpenCogMacros")

# C++ unit testing
FIND_PACKAGE(Cxxtest)
IF (CXXTEST_FOUND)
	MESSAGE(STATUS "CxxTest found.")
ELSE (CXXTEST_FOUND)
	MESSAGE(STATUS "CxxTest missing: needed for unit tests.")
ENDIF (CXXTEST_FOUND)

# ===================================================================
# Build subdirectories
ADD_SUBDIRECTORY(opencog)

# TODO: replace by include from cogutil
IF (CXXTEST_FOUND)
	ADD_CUSTOM_TARGET(tests)
	ADD_SUBDIRECTORY(tests EXCLUDE_FROM_ALL)
	ADD_CUSTOM_TARGET(test
		DEPENDS tests
		WORKING_DIRECTORY tests
		COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --output-on-failure $(ARGS)
		COMMENT "Running tests..."
	)
ENDIF (CXXTEST_FOUND)

# ===================================================================
# Show a summary of what we found, what we will do.

SUMMARY_ADD("CogUtil" "OpenCog utilities" COGUTIL_FOUND)
SUMMARY_ADD("AtomSpace" "Base atoms definitions" ATOMSPACE_FOUND)
SUMMARY_ADD("Cxxtest" "C++ unit testing" CXXTEST_FOUND)
SUMMARY_ADD("Cython bindings" "Cython (python) bindings" HAVE_CYTHON)
SUMMARY_ADD("Python tests" "Python bindings nose tests" HAVE_NOSETESTS)

SUMMARY_SHOW()
