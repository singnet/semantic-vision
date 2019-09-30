rm *pb2.py
rm *pb2_grpc.py
protoc --grpc_out=./ --plugin=protoc-gen-grpc=/usr/local/bin/grpc_python_plugin MinecraftizingService.proto
protoc --python_out=./ MinecraftizingService.proto
