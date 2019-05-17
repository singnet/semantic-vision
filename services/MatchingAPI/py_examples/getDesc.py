import grpc

import MatchingAPI_pb2
import MatchingAPI_pb2_grpc

def main():
    image_path_1 = '../Woods.jpg'
    detector = 'ORB'
    detector_params = ''
    descriptor = 'ORB'
    descriptor_params = ''
    channel = grpc.insecure_channel('localhost:50051', options=[('grpc.max_send_message_length', -1), (
        'grpc.max_receive_message_length', -1)])
    service = MatchingAPI_pb2_grpc.MatchApiStub(channel)
    request = MatchingAPI_pb2.descriptorRequest()
    with open(image_path_1, 'rb') as f:
        request.image = f.read()
    request.detector_name = detector
    request.det_parameters = detector_params
    request.descriptor_name = descriptor
    request.desc_parameters = descriptor_params
    response = service.getDescByImage(request)
    print(response.status)


if __name__ == "__main__":
    main()