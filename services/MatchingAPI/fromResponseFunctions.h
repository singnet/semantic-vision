#ifndef MATCHINGAPI_FROMRESPONSEFUNCTIONS_H
#define MATCHINGAPI_FROMRESPONSEFUNCTIONS_H

#include "Includes.h"

#include "MatchingAPI.grpc.pb.h"

using MatchingApi::descriptorByKpsRequest;
using MatchingApi::descriptorResponse;
using MatchingApi::matchingRequest;
using MatchingApi::keypointResponse;
using MatchingApi::matchingByImageResponse;
using MatchingApi::transformRequest;

template <class T> vector<KeyPoint> kpsFromResponse(T request)
{
    vector<KeyPoint> result;
    for (auto& kp : request)
    {
        result.emplace_back(kp.x(), kp.y(),kp.size(), kp.angle(), kp.response(), kp.octave(), kp.class_id());
    }
    return result;
}
void descsFromMatchRequest(matchingRequest request, Mat * desc1, Mat * desc2);
void matchesFromRequest(transformRequest request, vector<DMatch> * matches);

#endif //MATCHINGAPI_FROMRESPONSEFUNCTIONS_H
