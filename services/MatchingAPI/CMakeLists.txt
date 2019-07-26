cmake_minimum_required(VERSION 3.10)
project(MatchingAPI)

find_package(OpenCV 4.0 REQUIRED)
find_package(Protobuf REQUIRED)

if(NOT PYTHON_EXECUTABLE)
    if(NumPy_FIND_QUIETLY)
        find_package(PythonInterp 3.6 QUIET)
    else()
        find_package(PythonInterp 3.6)
        set(__numpy_out 1)
    endif()
endif()

if (PYTHON_EXECUTABLE)
    # Find out the include path
    execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c
            "from __future__ import print_function\ntry: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
            OUTPUT_VARIABLE __numpy_path)
    # And the version
    execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c
            "from __future__ import print_function\ntry: import numpy; print(numpy.__version__, end='')\nexcept:pass\n"
            OUTPUT_VARIABLE __numpy_version)
elseif(__numpy_out)
    message(STATUS "Python executable not found.")
endif(PYTHON_EXECUTABLE)

find_path(PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
        HINTS "${__numpy_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)

if(PYTHON_NUMPY_INCLUDE_DIR)
    set(PYTHON_NUMPY_FOUND 1 CACHE INTERNAL "Python numpy found")
endif(PYTHON_NUMPY_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy REQUIRED_VARS PYTHON_NUMPY_INCLUDE_DIR
        VERSION_VAR __numpy_version)

message("Python numpy include dir " ${PYTHON_NUMPY_INCLUDE_DIR})
find_package(PythonLibs REQUIRED)
message("Python include dir " ${PYTHON_INCLUDE_DIR})

find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

set(CMAKE_CXX_STANDARD 17)
SET(GCC_COVERAGE_LINK_FLAG "-lprotobuf -pthread")

add_executable(Client Client.cpp MatchingAPI.grpc.pb.h MatchingAPI.pb.cc MatchingAPI.pb.h MatchingAPI.grpc.pb.cc)

add_executable(Server Server.cpp MatchingAPI.grpc.pb.h MatchingAPI.pb.cc MatchingAPI.pb.h MatchingAPI.grpc.pb.cc
        DetectInterface.h FeaturesInterface.h Includes.h Detect.cpp Features.cpp CheckParam.h
        argvParser.cpp argvParser.h get_functions.cpp get_functions.h fromResponseFunctions.cpp fromResponseFunctions.h
        Transformations.cpp Transformations.h additionalTransformations.cpp additionalTransformations.h
        additionalFunctionForTransforms.cpp additionalFunctionForTransforms.h getBOW.h getBOW.cpp)

include_directories(${PROJECT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRECTORIES})
target_link_libraries(Client ${OpenCV_LIBS} ${Protobuf_LIBRARIES} grpc++_unsecure grpc gpr)
target_link_libraries(Server ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Protobuf_LIBRARIES} grpc++_unsecure grpc gpr)
