cmake_minimum_required(VERSION 3.10)
project(MatchingAPI)

find_package(OpenCV 4.0 REQUIRED)
#message("OpenCV install path " ${OpenCV_INSTALL_PATH})
find_package(Protobuf REQUIRED)
#message("Protobuf include dirs " ${Protobuf_INCLUDE_DIRS})

#set(PYTHON_LIBRARY "/home/daddywesker/anaconda3/envs/pgconda/lib/libpython3.6m.so")
#set(PYTHON_INCLUDE_DIR "/home/daddywesker/anaconda3/envs/pgconda/include/python3.6m")

find_package(PythonLibs REQUIRED)
#message("Python include dir " ${PYTHON_INCLUDE_DIR})

find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
#message("GRPC cpp plugin path " ${GRPC_CPP_PLUGIN})

set(CMAKE_CXX_STANDARD 14)
SET(GCC_COVERAGE_LINK_FLAG "-lprotobuf -pthread")

add_executable(Client Client.cpp MatchingAPI.grpc.pb.h MatchingAPI.pb.cc MatchingAPI.pb.h MatchingAPI.grpc.pb.cc)

add_executable(Server Server.cpp MatchingAPI.grpc.pb.h MatchingAPI.pb.cc MatchingAPI.pb.h MatchingAPI.grpc.pb.cc
        DetectInterface.h FeaturesInterface.h Includes.h Detect.cpp Features.cpp CheckParam.h
        argvParser.cpp argvParser.h get_functions.cpp get_functions.h fromResponseFunctions.cpp fromResponseFunctions.h
        Transformations.cpp Transformations.h additionalTransformations.cpp additionalTransformations.h
        additionalFunctionForTransforms.cpp additionalFunctionForTransforms.h)


#set(PYTHON_INCLUDE_PATH "$ENV{HOME}/anaconda3/envs/py3.6")
#
#set(PYTHON_LIBRARIES "$ENV{HOME}/anaconda3/envs/py3.6/lib/libpython3.6m.so")

include_directories(${PROJECT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRECTORIES})
#include_directories(${PATH})
target_link_libraries(Client ${OpenCV_LIBS} ${Protobuf_LIBRARIES} grpc++_unsecure grpc gpr)
target_link_libraries(Server ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Protobuf_LIBRARIES} grpc++_unsecure grpc gpr)

#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#    message(STATUS "dir='${dir}'")
#endforeach()
