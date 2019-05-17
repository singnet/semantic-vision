import grpc

import MatchingAPI_pb2
import MatchingAPI_pb2_grpc

import numpy as np
import cv2

import matplotlib.pyplot as plt

def keypointsFromResponseKP(in_keypoints):
    keypoints = []
    for oneKP in in_keypoints:
        keypoints.append(cv2.KeyPoint(oneKP.x, oneKP.y, oneKP.size, oneKP.angle, oneKP.response, oneKP.octave, oneKP.class_id))
    return keypoints

def matchesFromResponse(response):
    matches = []
    for oneMatch in response.all_matches:
        matches.append(cv2.DMatch(oneMatch.queryIdx, oneMatch.trainIdx, oneMatch.imgIdx, oneMatch.distance))
    return matches

def main():
    image_path_1 = '../Woods.jpg'
    image_path_2 = '../Woods2.jpg'
    detector = 'ORB'
    detector_params = 'nfeatures 1000 scaleFactor 5'
    descriptor = 'ORB'
    descriptor_params = ''
    channel = grpc.insecure_channel('localhost:50051', options=[('grpc.max_send_message_length', -1), (
        'grpc.max_receive_message_length', -1)])
    service = MatchingAPI_pb2_grpc.MatchApiStub(channel)
    request = MatchingAPI_pb2.matchingByImageRequest()
    with open(image_path_1, 'rb') as f:
        request.image_first = f.read()
    with open(image_path_2, 'rb') as f:
        request.image_second = f.read()
    request.detector_name = detector
    request.det_parameters = detector_params
    request.descriptor_name = descriptor
    request.desc_parameters = descriptor_params
    response = service.getMatchByImage(request)
    print(response.status)

    imageMat1 = cv2.imread(image_path_1)
    imageMat2 = cv2.imread(image_path_2)

    kps1 = keypointsFromResponseKP(response.keypoints_first)
    kps2 = keypointsFromResponseKP(response.keypoints_second)
    matches_b_img = matchesFromResponse(response)

    image_matches = np.empty((max(imageMat1.shape[0], imageMat2.shape[0]), imageMat1.shape[1] + imageMat2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(imageMat1, kps1, imageMat2, kps2, matches_b_img, image_matches, flags=2)
    plt.imsave("Match.jpg",image_matches)


if __name__ == "__main__":
    main()