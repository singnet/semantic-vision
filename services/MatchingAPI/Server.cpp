#include <grpc/grpc.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc++/server_context.h>
#include <grpc++/security/server_credentials.h>
#include "MatchingAPI.grpc.pb.h"

#include <Python.h>

#include "Includes.h"
#include "get_functions.h"
#include "fromResponseFunctions.h"

#include <csignal>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
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
using MatchingApi::MatchApi;

static void fillKeypoint(MatchingApi::keyPoint* kp, KeyPoint oneVec)
{
    kp->set_angle(oneVec.angle);
    kp->set_class_id(oneVec.class_id);
    kp->set_octave(oneVec.octave);
    kp->set_response(oneVec.response);
    kp->set_size(oneVec.size);
    kp->set_x(oneVec.pt.x);
    kp->set_y(oneVec.pt.y);
}

template <class T> static void fillFeaturesF(const vector<vector<float>>* featuresVec, T* reply)
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

template <class T> static void fillFeaturesU(const vector<vector<int>>* featuresVec, T* reply)
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

template <class T> static void fillMatches(const vector<DMatch>* matches, T* reply)
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

static void signalHandler( int signum ) {
    cout << "Interrupt signal (" << signum << ") received.\n";
    exit(signum);
}


class MatchingApiServer final : public MatchApi::Service {
    Status getKP(ServerContext* context, const keypointRequest* request,
                    keypointResponse* reply) override {
        vector<KeyPoint> keypointsFromMAPI;
        string result = getKeypoint(request->image(), request->detector_name(), request->parameters(), &keypointsFromMAPI);
        if (strcmp(result.c_str(), "Zero_keypoints_detected") == 0) {
            (*reply).set_status(result);
            return Status::OK;
        }
        cout << endl;
        for (auto& oneVec : keypointsFromMAPI)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints();
            fillKeypoint(buf, oneVec);
        }
        reply->set_status(result);
        return Status::OK;

    }

    Status getDescByImage(ServerContext* context, const descriptorRequest* request,
                   descriptorResponse* reply) override
    {
        vector<KeyPoint> keypointsFromMAPI;
        vector<vector<float>> featuresF;
        vector<vector<int>> featuresU;
        string result = getDescriptorByImage(request->image(), request->detector_name(),
                                                      request->det_parameters(),
                                                      request->descriptor_name(), request->desc_parameters(),
                                                      &featuresF, &featuresU, &keypointsFromMAPI);

        cout << endl;

        if (strcmp(result.c_str(), "Zero_keypoints_detected") == 0) {
            (*reply).set_status(result);
            return Status::CANCELLED;
        }

        fillFeaturesF(&featuresF,reply);
        fillFeaturesU(&featuresU,reply);

        for (auto& oneVec : keypointsFromMAPI)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints();
            fillKeypoint(buf, oneVec);
        }
        reply->set_status(result);
        return Status::OK;
    }

    Status getDescByKps (ServerContext* context, const descriptorByKpsRequest* request,
                         descriptorResponse* reply) override
    {
        vector<vector<float>> featuresF;
        vector<vector<int>> featuresU;
        vector<KeyPoint> keypointsFromMAPI = kpsFromResponse((*request).keypoints());

        string result = getDescriptorByKps(request->image(), request->descriptor_name(),
                request->desc_parameters(), keypointsFromMAPI, &featuresF, &featuresU);

        cout << endl;
        fillFeaturesF(&featuresF,reply);
        fillFeaturesU(&featuresU,reply);
        for (auto& oneVec : keypointsFromMAPI)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints();
            fillKeypoint(buf, oneVec);
        }
        reply->set_status(result);
        return Status::OK;
    }

    Status getMatch(ServerContext* context, const matchingRequest* request,
                    matchingResponse* reply) override
    {
        vector<DMatch> matches;
        Mat desc1, desc2;
        descsFromMatchRequest((*request), &desc1, &desc2);
        string result = getMatches(desc1, desc2, &matches);
        cout << endl;
        fillMatches(&matches, reply);
        reply->set_status(result);
        return Status::OK;
    }

    Status getMatchByImage(ServerContext* context, const matchingByImageRequest* request, matchingByImageResponse* reply) override
    {
        vector<KeyPoint> kps1;
        vector<KeyPoint> kps2;
        vector<DMatch> matches;
        string result = getMatchesByImg(request->image_first(), request->image_second(), request->detector_name(), request->det_parameters(),
                request->descriptor_name(), request->desc_parameters(), &kps1, &kps2, &matches);

        fillMatches(&matches, reply);

        for (auto& oneVec : kps1)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints_first();
            fillKeypoint(buf, oneVec);
        }

        for (auto& oneVec : kps2)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints_second();
            fillKeypoint(buf, oneVec);
        }

        reply->set_status(result);
        return Status::OK;
    }

    Status getTransformParameters(ServerContext* context, const transformRequest* request, transformResponse* reply) override
    {
        vector<DMatch> matches_in;
        vector<KeyPoint> first_kps = kpsFromResponse(request->keypoints_first());
        vector<KeyPoint> second_kps = kpsFromResponse(request->keypoints_second());
        matchesFromRequest((*request), &matches_in);
        vector<double> transform_parameters;
        string result = getTransformParams(request->transform_type(), request->transform_input_parameters(), matches_in, first_kps, second_kps, &transform_parameters);

        for (auto& oneParam : transform_parameters)
        {
            reply->add_transform_parameters(oneParam);
        }
        cout << endl;
        reply->set_status(result);
        return Status::OK;
    }

    Status getTransformParametersByImage(ServerContext* context, const transformByImageRequest* request, transformResponse* reply) override
    {
        vector<double> transform_parameters;
        string result = getTransformParamsByImg(request->image_first(), request->image_second(), request->detector_name(),
                request->det_parameters(), request->descriptor_name(), request->desc_parameters(), request->transform_type(),
                request->transform_input_parameters(), &transform_parameters);


        for (auto& oneParam : transform_parameters)
        {
            reply->add_transform_parameters(oneParam);
        }
        reply->set_status(result);
        return Status::OK;
    }

};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    MatchingApiServer service;

    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}

int main(int argc, char** argv) {
    setenv("PYTHONPATH", "../models/", 1);
    signal(SIGINT, signalHandler);
    Py_InitializeEx(0);
    RunServer();
    Py_FinalizeEx();
    return 0;
}