import grpc

import MatchingAPI_pb2
import MatchingAPI_pb2_grpc

def main():
    image_path_1 = '../Woods.jpg'
    detector = 'ORB'
    detector_params = ''
    channel = grpc.insecure_channel('localhost:50051', options=[('grpc.max_send_message_length', -1), (
        'grpc.max_receive_message_length', -1)])
    service = MatchingAPI_pb2_grpc.MatchApiStub(channel)
    request = MatchingAPI_pb2.keypointRequest()
    with open(image_path_1, 'rb') as f:
        request.image = f.read()
    request.detector_name = detector
    request.parameters = detector_params
    response = service.getKP(request)
    print(response.status)
    print("num of keypoints")
    print(len(response.keypoints))


if __name__ == "__main__":
    main()
