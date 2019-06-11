cmake_minimum_required(VERSION 3.10)
project(MatchingAPI)
find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)

find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

set(CMAKE_CXX_STANDARD 14)
SET(GCC_COVERAGE_LINK_FLAG "-lprotobuf -pthread")

add_executable(Client Client.cpp MatchingAPI.grpc.pb.h MatchingAPI.pb.cc MatchingAPI.pb.h MatchingAPI.grpc.pb.cc
        fromResponseFunctions.cpp fromResponseFunctions.h)

add_executable(Server Server.cpp MatchingAPI.grpc.pb.h MatchingAPI.pb.cc MatchingAPI.pb.h MatchingAPI.grpc.pb.cc
        DetectInterface.h FeaturesInterface.h Includes.h Detect.cpp Features.cpp CheckParam.h
        argvParser.cpp argvParser.h get_functions.cpp get_functions.h fromResponseFunctions.cpp fromResponseFunctions.h Transformations.cpp Transformations.h additionalTransformations.cpp additionalTransformations.h additionalFunctionForTransforms.cpp additionalFunctionForTransforms.h)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PATH})
target_link_libraries(Client ${OpenCV_LIBS} ${Protobuf_LIBRARIES} grpc++_unsecure grpc gpr)
target_link_libraries(Server ${OpenCV_LIBS} ${Protobuf_LIBRARIES} grpc++_unsecure grpc gpr)