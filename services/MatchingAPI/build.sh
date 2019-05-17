protoc --grpc_out=py_examples/ --plugin=protoc-gen-grpc=/usr/local/bin/grpc_python_plugin MatchingAPI.proto
protoc --python_out=py_examples/ MatchingAPI.proto
protoc --grpc_out=./ --plugin=protoc-gen-grpc=/usr/local/bin/grpc_cpp_plugin MatchingAPI.proto
protoc --cpp_out=./ MatchingAPI.proto
mkdir build
cd build
cmake ..
make
