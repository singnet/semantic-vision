# ===================================================================
# Build system settings

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(opencog-neuralnet VERSION 0.0.1)

# set default build type
IF (CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Release)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "")

# DATADIR is required by OpenCogMacros.cmake
IF (NOT DEFINED DATADIR)
	SET (DATADIR "${CMAKE_INSTALL_PREFIX}/share/opencog")
ENDIF (NOT DEFINED DATADIR)
ADD_DEFINITIONS(-DDATADIR="${DATADIR}")

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

INCLUDE(${COGUTIL_DATA_DIR}/cmake/Summary.cmake)

# AtomSpace
FIND_PACKAGE(AtomSpace 5.0.4 CONFIG REQUIRED)
IF (ATOMSPACE_FOUND)
	MESSAGE(STATUS "AtomSpace found.")
ELSE (ATOMSPACE_FOUND)
	MESSAGE(FATAL_ERROR "AtomSpace missing: it is needed!")
ENDIF (ATOMSPACE_FOUND)

INCLUDE("${ATOMSPACE_DATA_DIR}/cmake/OpenCogMacros.cmake")

# ===================================================================
# Build subdirectories
ADD_SUBDIRECTORY(opencog)

# ===================================================================
# Show a summary of what we found, what we will do.

SUMMARY_ADD("CogUtil" "OpenCog utilities" HAVE_COGUTIL)
SUMMARY_ADD("AtomSpace" "Base atoms definitions" HAVE_ATOMSPACE)
SUMMARY_ADD("Cython bindings" "Cython (python) bindings" HAVE_CYTHON)
SUMMARY_ADD("Python tests" "Python bindings nose tests" HAVE_NOSETESTS)

SUMMARY_SHOW()
