#ifndef MATCHINGAPI_UTILITIES_H
#define MATCHINGAPI_UTILITIES_H

#include "Includes.h"

Mat getMat(string imageBytes);
string convertImgToString (Mat img);
string getImageString(string path);

template <class T, class F> void fillKeypoint(T cv_kp, F* msg_kp)
{
    msg_kp->set_angle(cv_kp.angle);
    msg_kp->set_class_id(cv_kp.class_id);
    msg_kp->set_octave(cv_kp.octave);
    msg_kp->set_response(cv_kp.response);
    msg_kp->set_size(cv_kp.size);
    msg_kp->set_x(cv_kp.pt.x);
    msg_kp->set_y(cv_kp.pt.y);
}

template <class T> void fillFeaturesF(const vector<vector<float>>* featuresVec, T* reply)
{
    for (auto& oneVec : (*featuresVec))
    {
        MatchingApi::oneDescriptor* buf = reply->add_features();
        for (auto& oneValue : oneVec)
        {
            buf->add_onedescf(oneValue);
        }
    }
}

template <class T> void fillFeaturesU(const vector<vector<int>>* featuresVec, T* reply)
{
    for (auto& oneVec : (*featuresVec))
    {
        MatchingApi::oneDescriptor* buf = reply->add_features();
        for (auto& oneValue : oneVec)
        {
            buf->add_onedescu(oneValue);
        }
    }
}

template <class T> void fillMatches(const vector<DMatch>* matches, T* reply)
{
    for (auto& oneMatch : (*matches))
    {
        MatchingApi::matchedPoint* buf = reply->add_all_matches();
        buf->set_distance(oneMatch.distance);
        buf->set_imgidx(oneMatch.imgIdx);
        buf->set_queryidx(oneMatch.queryIdx);
        buf->set_trainidx(oneMatch.trainIdx);
    }
}

template <class T, class F> void fillDescF(F* oneFeature, T* buf)
{
    for (auto& oneDesc : (*oneFeature).onedescf())
        buf->add_onedescf(oneDesc);
}

template <class T, class F> void fillDescU(F* oneFeature, T* buf)
{
    for (auto& oneDesc : (*oneFeature).onedescu())
        buf->add_onedescu(oneDesc);
}

#endif //MATCHINGAPI_UTILITIES_H
