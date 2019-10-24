#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc++/server_context.h>
#include <grpc++/security/server_credentials.h>

#include <Python.h>

#include "get_functions.h"
#include "fromResponseFunctions.h"

#include <csignal>

#include <random>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;

static void signalHandler( int signum ) {
    cout << "Interrupt signal (" << signum << ") received.\n";
    exit(signum);
}


class MatchingApiServer final : public MatchApi::Service {
    Status getKP(ServerContext* context, const keypointRequest* request,
                    keypointResponse* reply) override {
        vector<KeyPoint> keypointsFromMAPI;
        string result = getKeypoint(request->image(), request->detector_name(), request->parameters(), &keypointsFromMAPI);
        if (strcmp(result.c_str(), "Success") != 0) {
            (*reply).set_status(result);
            return Status::CANCELLED;
        }
        cout << endl;
        for (auto& oneVec : keypointsFromMAPI)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints();
            fillKeypoint(oneVec, buf);
        }
        reply->set_status(result);
        Mat in = getMat(request->image());
        int type = in.type();
        Mat imageKP;
        try
        {
            drawKeypoints(in, keypointsFromMAPI, imageKP);
        }
        catch (const std::exception &exc)
        {
            cout << exc.what() << endl;
        }

        string encoded = convertImgToString(imageKP);
        reply->set_uiimage(encoded);
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

        if (strcmp(result.c_str(), "Success") != 0) {
            (*reply).set_status(result);
            Status status(grpc::INVALID_ARGUMENT, result);
            return status;
        }

        fillFeaturesF(&featuresF,reply);
        fillFeaturesU(&featuresU,reply);

        for (auto& oneVec : keypointsFromMAPI)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints();
            fillKeypoint(oneVec, buf);
        }
        reply->set_status(result);
        return Status::OK;
    }

    Status getDescByKps (ServerContext* context, const descriptorByKpsRequest* request,
                         descriptorResponse* reply) override
    {
        vector<vector<float>> featuresF;
        vector<vector<int>> featuresU;
        if ((*request).keypoints_size() == 0)
        {
            (*reply).set_status("No keypoints given");
            Status status(grpc::INVALID_ARGUMENT, "No keypoints given");
            return status;
        }
        vector<KeyPoint> keypointsFromMAPI = kpsFromResponse((*request).keypoints());

        string result = getDescriptorByKps(request->image(), request->descriptor_name(),
                request->desc_parameters(), keypointsFromMAPI, &featuresF, &featuresU);

        if (strcmp(result.c_str(), "Success") != 0) {
            (*reply).set_status(result);
            Status status(grpc::INVALID_ARGUMENT, result);
            return status;
        }

        fillFeaturesF(&featuresF,reply);
        fillFeaturesU(&featuresU,reply);
        for (auto& oneVec : keypointsFromMAPI)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints();
            fillKeypoint(oneVec, buf);
        }
        reply->set_status(result);
        return Status::OK;
    }

    Status getMatch(ServerContext* context, const matchingRequest* request,
                    matchingResponse* reply) override
    {
        vector<DMatch> matches;
        if (request->features_first_size() == 0)
        {
            (*reply).set_status("No features given for the first image");
            Status status(grpc::INVALID_ARGUMENT, "No features given for the first image");
            return status;
        }
        if (request->features_second_size() == 0)
        {
            (*reply).set_status("No features given for the second image");
            Status status(grpc::INVALID_ARGUMENT, "No features given for the second image");
            return status;
        }
        Mat desc1, desc2;
        descsFromMatchRequest((*request), &desc1, &desc2);
        string result = getMatches(desc1, desc2, &matches);
        cout << endl;
        if (strcmp(result.c_str(), "Success") != 0) {
            (*reply).set_status(result);
            Status status(grpc::INVALID_ARGUMENT, result);
            return status;
        }
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

        if (strcmp(result.c_str(), "Success") != 0) {
            (*reply).set_status(result);
            Status status(grpc::INVALID_ARGUMENT, result);
            return status;
        }
        Mat firstImg = getMat(request->image_first());
        Mat secondImg = getMat(request->image_second());
        Mat matchImage;
        drawMatches(firstImg, kps1, secondImg, kps2, matches, matchImage);
        fillMatches(&matches, reply);
        string encoded = convertImgToString(matchImage);
        reply->set_uiimage(encoded);

        for (auto& oneVec : kps1)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints_first();
            fillKeypoint(oneVec, buf);
        }

        for (auto& oneVec : kps2)
        {
            MatchingApi::keyPoint* buf = reply->add_keypoints_second();
            fillKeypoint(oneVec, buf);
        }

        reply->set_status(result);
        return Status::OK;
    }

    Status getTransformParameters(ServerContext* context, const transformRequest* request, transformResponse* reply) override
    {
        vector<DMatch> matches_in;
        if (request->keypoints_first_size() == 0)
        {
            (*reply).set_status("No keypoints given for the first image");
            Status status(grpc::INVALID_ARGUMENT, "No keypoints given for the first image");
            return status;
        }
        if (request->keypoints_second_size() == 0)
        {
            (*reply).set_status("No keypoints given for the second image");
            Status status(grpc::INVALID_ARGUMENT, "No keypoints given for the second image");
            return status;
        }
        if (request->all_matches_size() == 0)
        {
            (*reply).set_status("No matches given");
            Status status(grpc::INVALID_ARGUMENT, "No matches given");
            return status;
        }
        vector<KeyPoint> first_kps = kpsFromResponse(request->keypoints_first());
        vector<KeyPoint> second_kps = kpsFromResponse(request->keypoints_second());
        matchesFromRequest((*request), &matches_in);
        vector<double> transform_parameters;
        string result = getTransformParams(request->transform_type(), request->transform_input_parameters(), matches_in, first_kps, second_kps, &transform_parameters);
        cout << endl;
        if (strcmp(result.c_str(), "Success") != 0) {
            (*reply).set_status(result);
            Status status(grpc::INVALID_ARGUMENT, result);
            return status;
        }
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
        Mat resImage, mixedImage;
        string result = getTransformParamsByImg(request->image_first(), request->image_second(), request->detector_name(),
                request->det_parameters(), request->descriptor_name(), request->desc_parameters(), request->transform_type(),
                request->transform_input_parameters(), &transform_parameters, resImage, mixedImage);
        int rnd = rand();
        string encoded = convertImgToString(resImage);
        string encodedMixed = convertImgToString(mixedImage);
        reply->set_uiimage(encodedMixed);
        reply->set_resultimage(encoded);
        cout << endl;
        if (strcmp(result.c_str(), "Success") != 0) {
            (*reply).set_status(result);
            Status status(grpc::INVALID_ARGUMENT, result);
            return status;
        }
        for (auto& oneParam : transform_parameters)
        {
            reply->add_transform_parameters(oneParam);
        }
        reply->set_status(result);
        return Status::OK;
    }

    Status getClosestImages(ServerContext* context, const imageRetrievalRequest* request, imageRetrievalResponse* reply) override
    {
        cout << "entered get function" << endl;
        vector<float> distances;
        vector<string> retrievedImages;
        vector<string> dataBase;
        for (auto& oneImg : request->image_base())
        {
            dataBase.push_back(oneImg);
        }
        string result = getClosestImg(request->input_image(), dataBase, request->descriptor_name(), request->desc_parameters(),
                request->detector_name(), request->det_parameters(), request->numofclusters(), request->numofimagestoretrieve(), &retrievedImages,
                &distances);
        cout << endl;
        if (strcmp(result.c_str(), "Success") != 0) {
            (*reply).set_status(result);
            Status status(grpc::INVALID_ARGUMENT, result);
            return status;
        }
        reply->set_status(result);
        for (auto& distance : distances)
        {
            reply->add_distances(distance);
        }
        for (auto& rImage: retrievedImages)
        {
            reply->add_images(rImage);
        }
        if (reply->distances_size() != 0)
        {
            Mat concatted, concattedPre;
            Mat original = getMat(request->input_image());
            concattedPre = original.clone();
            int width = 640;
            int height = 480;
            resize(concattedPre, concatted, Size(width, height), 0, 0, INTER_CUBIC);
            string distance = "Distance ";
            for (auto &oneImage : retrievedImages)
            {
                Mat buf = getMat(oneImage);
                Mat buf2;
                resize(buf, buf2, Size(width, height), 0, 0, INTER_CUBIC);
                hconcat(concatted, buf2, concatted);
            }
            std::string out_string;
            std::stringstream ss;
            out_string = ss.str();
            imwrite("IR_result.png", concatted);
            string encodedMixed = convertImgToString(concatted);
            reply->set_uiimage(encodedMixed);
            cout << endl;
        }
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
