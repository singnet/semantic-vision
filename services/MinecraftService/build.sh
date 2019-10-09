rm *pb2.py
rm *pb2_grpc.py
protoc --grpc_out=./ --plugin=protoc-gen-grpc=/usr/local/bin/grpc_python_plugin MinecraftizingService.proto
protoc --python_out=./ MinecraftizingService.proto
wget --no-check-certificate https://snet-models.s3.amazonaws.com/semantic-vision/MinecraftService/CycleGan.tar
tar -xvf CycleGan.tar
rm CycleGan.tar
wget --no-check-certificate https://snet-models.s3.amazonaws.com/semantic-vision/MinecraftService/UGATIT.tar
tar -xvf UGATIT.tar
rm UGATIT.tar
