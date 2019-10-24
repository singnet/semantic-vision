#ifndef MATCHINGAPI_INCLUDES_H
#define MATCHINGAPI_INCLUDES_H

#include <grpc/grpc.h>
#include "MatchingAPI.grpc.pb.h"

using grpc::Status;
using MatchingApi::descriptorRequest;
using MatchingApi::descriptorResponse;
using MatchingApi::keypointRequest;
using MatchingApi::keypointResponse;
using MatchingApi::descriptorByKpsRequest;
using MatchingApi::matchingRequest;
using MatchingApi::matchingResponse;
using MatchingApi::matchingByImageRequest;
using MatchingApi::matchingByImageResponse;
using MatchingApi::transformRequest;
using MatchingApi::transformResponse;
using MatchingApi::transformByImageRequest;
using MatchingApi::imageRetrievalResponse;
using MatchingApi::imageRetrievalRequest;
using MatchingApi::MatchApi;

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "base64.h"

#include <dirent.h>

using namespace cv;
using namespace std;
using namespace xfeatures2d;

#endif