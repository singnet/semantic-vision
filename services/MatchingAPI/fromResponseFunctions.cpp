#include "fromResponseFunctions.h"

Mat matFromFeatures(google::protobuf::RepeatedPtrField<MatchingApi::oneDescriptor> features)
{
    Mat desc;
    const MatchingApi::oneDescriptor* const* ptrFeatures = features.data();
    if ((*ptrFeatures)->onedescu().size() > 0)
    {
        for (auto& oneFeature : features)
        {
            google::protobuf::RepeatedField<int> ptr_onedescu = oneFeature.onedescu();
            vector<int> int_desc(ptr_onedescu.data(), ptr_onedescu.data() + ptr_onedescu.size());
            vector<uchar> buf;
            for (auto& oneFeat : int_desc)
            {
                buf.push_back((uchar)oneFeat);
            }
            Mat bufMat(buf);
            desc.push_back(bufMat.reshape(1, 1));
        }
    }
    else if ((*ptrFeatures)->onedescf().size() > 0)
    {
        for (auto& oneFeature : features)
        {
            google::protobuf::RepeatedField<float> ptr_onedescf = oneFeature.onedescf();
            vector<float> buf(ptr_onedescf.data(), ptr_onedescf.data() + ptr_onedescf.size());
            Mat bufMat(buf);
            desc.push_back(bufMat.reshape(1, 1));
        }
    }
    return desc;
}

void descsFromMatchRequest(matchingRequest request, Mat * desc1, Mat * desc2)
{
    (*desc1) = matFromFeatures(request.features_first());
    (*desc2) = matFromFeatures(request.features_second());
}

void matchesFromRequest(transformRequest request, vector<DMatch> * matches)
{
    for (auto& oneMatch: request.all_matches())
    {
        DMatch buf;
        buf.trainIdx = oneMatch.trainidx();
        buf.queryIdx = oneMatch.queryidx();
        buf.imgIdx = oneMatch.imgidx();
        buf.distance = oneMatch.distance();
        (*matches).push_back(buf);
    }
}

void matchesFromAllMatches (google::protobuf::RepeatedPtrField< ::MatchingApi::matchedPoint > requestMatches, vector<DMatch> &matches)
{
    for (auto& oneMatch: requestMatches)
    {
        DMatch buf;
        buf.trainIdx = oneMatch.trainidx();
        buf.queryIdx = oneMatch.queryidx();
        buf.imgIdx = oneMatch.imgidx();
        buf.distance = oneMatch.distance();
        matches.push_back(buf);
    }
}