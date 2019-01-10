import grpc

import service_pb2
import service_pb2_grpc

def main():
    channel = grpc.insecure_channel('localhost:12345')
    service = service_pb2_grpc.VqaServiceStub(channel)
    request = service_pb2.VqaRequest()
    request.question = "How many zebras are there?"
    request.use_pm = True
    request.image_format = 'jpg'
    with open('/home/relex/projects/data/coco/COCO_val2014_000000999999.jpg', 'rb') as f:
        request.data = f.read()
    response = service.answer(request)
    print(response)

if __name__ == "__main__":
    main()
