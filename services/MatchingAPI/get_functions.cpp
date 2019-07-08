#include "get_functions.h"

#include "DetectInterface.h"
#include "FeaturesInterface.h"
#include "Transformations.h"
#include "argvParser.h"

#include <fstream>

static string descToVec(vector<vector<float>> *resultF,
            vector<vector<int>> *resultU, Mat desc_result)
{
    int type = desc_result.type();

    if (type == 0)
    {
        for (int i = 0; i < desc_result.rows; i++)
        {
            const uchar *p = desc_result.ptr<uchar>(i);
            vector<int> buf;
            for (int j = 0; j < desc_result.cols; j++)
            {
                buf.push_back((int)p[j]);
            }
            (*resultU).push_back(buf);
        }
        return "Success. Features' type uchar";
    }
    else if (type == 5)
    {
        for (int i = 0; i < desc_result.rows; i++)
        {
            const float *p = desc_result.ptr<float>(i);
            (*resultF).emplace_back(p, p + desc_result.cols);
        }
        return "Success. Features' type float";
    }
}

static Mat getMat(string imageBytes)
{
    size_t length = imageBytes.size();
    Mat imageMat;
    vector<char> data((char*)imageBytes.c_str(), (char*)imageBytes.c_str() + length);
    imageMat = imdecode(data, IMREAD_UNCHANGED);
    return imageMat;
}

string getKeypoint(string image, string detector, string detector_parameters, vector<KeyPoint> * output)
{
    IDetector* ptr_detector = ChooseDetector(detector.c_str());
    map<string, double> params;
    parseClient(detector_parameters, &params);
    ptr_detector->setParameters(params);
    cout << endl;
    *output = ptr_detector->getPoints(image);
    ptr_detector->releaseDetector();
    if ((*output).size() == 0)
        return "Zero_keypoints_detected";
    return "Success";
}

string getDescriptorByImage(string image, string detector, string detector_parameters, string descriptor,
                                     string descriptor_parameters, vector<vector<float>> *resultF,
                                     vector<vector<int>> *resultU, vector<KeyPoint> *kps)
{
    IDescribe* ptr_describe = ChooseFeatures(descriptor.c_str());
    IDetector* ptr_detector = ChooseDetector(detector.c_str());

    map<string, double> det_params, desc_params;

    parseClient(detector_parameters, &det_params);
    parseClient(descriptor_parameters, &desc_params);

    cout << endl << "Setting parameters for detector:" << endl;
    ptr_detector->setParameters(det_params);
    cout << endl << endl << "Setting parameters for descriptor:" << endl;
    ptr_describe->setParameters(desc_params);

    (*kps) = ptr_detector->getPoints(image);
    ptr_detector->releaseDetector();
    if ((*kps).size() == 0) {
        ptr_describe->releaseDescriptor();
        return "Zero keypoints detected";
    }
    Mat desc_result = ptr_describe->getFeatures(kps, image);

    ptr_describe->releaseDescriptor();
    return descToVec(resultF, resultU, desc_result);
}

string getDescriptorByKps(string image, string descriptor, string descriptor_parameters, vector<KeyPoint> &kps,
                                   vector<vector<float>> *resultF, vector<vector<int>> *resultU)
{
    if (kps.size() == 0)
        return "You have given zero keypoints. No descriptors could be computed";
    Mat imageMat = getMat(image);
    IDescribe* ptr_describe = ChooseFeatures(descriptor.c_str());
    cout << endl << endl << "Setting parameters for descriptor:" << endl;
    map<string, double> det_params, desc_params;
    parseClient(descriptor_parameters, &desc_params);
    ptr_describe->setParameters(desc_params);

    Mat desc_result = ptr_describe->getFeatures(&kps, image);

    ptr_describe->releaseDescriptor();
    return descToVec(resultF, resultU, desc_result);
}

string getMatches(Mat desc1, Mat desc2, vector<DMatch>* matches)
{
    BFMatcher matcher(NORM_L2);
    matcher.match(desc1, desc2, (*matches));
    if ((*matches).size() == 0)
        return "No matches found";

    return "Matching done";
}

string getMatchesByImg(string image1, string image2, string detector, string detector_parameters, string descriptor,
        string descriptor_parameters, vector<KeyPoint> *kps1, vector<KeyPoint> *kps2, vector<DMatch>* matches)
{
    IDescribe* ptr_describe = ChooseFeatures(descriptor.c_str());
    IDetector* ptr_detector = ChooseDetector(detector.c_str());

    map<string, double> det_params, desc_params;

    parseClient(detector_parameters, &det_params);
    parseClient(descriptor_parameters, &desc_params);

    cout << endl << "Setting parameters for detector:" << endl;
    ptr_detector->setParameters(det_params);
    cout << endl << endl << "Setting parameters for descriptor:" << endl;
    ptr_describe->setParameters(desc_params);

    (*kps1) = ptr_detector->getPoints(image1);
    if ((*kps1).size() == 0) {
        ptr_describe->releaseDescriptor();
        ptr_detector->releaseDetector();
        return "Zero keypoints detected for the first image";
    }

    (*kps2) = ptr_detector->getPoints(image2);
    if ((*kps2).size() == 0) {
        ptr_describe->releaseDescriptor();
        ptr_detector->releaseDetector();
        return "Zero keypoints detected for the second image";
    }

    Mat desc_result1 = ptr_describe->getFeatures(kps1, image1);
    Mat desc_result2 = ptr_describe->getFeatures(kps2, image2);

    ptr_describe->releaseDescriptor();
    ptr_detector->releaseDetector();
    BFMatcher matcher(NORM_L2);
    matcher.match(desc_result1, desc_result2, (*matches));
    if ((*matches).size() == 0)
        return "No matches found";
    return "Matching done";
}

string getTransformParams(string transformType, string transform_input_parameters, vector<DMatch> matches_in, vector<KeyPoint> first_kps,
        vector<KeyPoint> second_kps, vector<double> * transform_parameters)
{
    sort(matches_in.begin(), matches_in.end(), [](DMatch a, DMatch b) { return a.distance < b.distance; });

    cout << endl << endl;
    ITransform* ptr_transform = ChooseTransform(transformType.c_str());

    map<string, double> transf_params;

    parseClient(transform_input_parameters, &transf_params);

    cout << endl << "Setting parameters for transform:" << endl;
    ptr_transform->setParameters(transf_params);

    (*transform_parameters) = ptr_transform->getTransform(first_kps, second_kps, matches_in);
    cout << endl;
    cout << (*transform_parameters).size() << " transform parameters as output" << endl;
    ptr_transform->releaseTransform();
    return "Transform parameters extraction is done";
}

string getTransformParamsByImg(string image1, string image2, string detector, string detector_parameters,
                               string descriptor, string descriptor_parameters, string transformType, string transform_input_parameters,
                               vector<double> * transform_parameters)
{
    IDescribe* ptr_describe = ChooseFeatures(descriptor.c_str());
    IDetector* ptr_detector = ChooseDetector(detector.c_str());
    ITransform* ptr_transform = ChooseTransform(transformType.c_str());

    map<string, double> det_params, desc_params, transf_params;
    vector<DMatch> matches;
    vector<KeyPoint> kps1, kps2;

    parseClient(detector_parameters, &det_params);
    parseClient(descriptor_parameters, &desc_params);
    parseClient(transform_input_parameters, &transf_params);

    cout << endl << "Setting parameters for detector:" << endl;
    ptr_detector->setParameters(det_params);
    cout << endl << endl << "Setting parameters for descriptor:" << endl;
    ptr_describe->setParameters(desc_params);
    cout << endl << "Setting parameters for transform:" << endl;
    ptr_transform->setParameters(transf_params);

    kps1 = ptr_detector->getPoints(image1);
    if (kps1.size() == 0) {
        ptr_describe->releaseDescriptor();
        ptr_detector->releaseDetector();
        ptr_transform->releaseTransform();
        return "Zero keypoints detected for the first image";
    }

    kps2 = ptr_detector->getPoints(image2);
    if (kps2.size() == 0) {
        ptr_describe->releaseDescriptor();
        ptr_detector->releaseDetector();
        ptr_transform->releaseTransform();
        return "Zero keypoints detected for the second image";
    }
    Mat desc_result1 = ptr_describe->getFeatures(&kps1, image1);
    Mat desc_result2 = ptr_describe->getFeatures(&kps2, image2);

    BFMatcher matcher(NORM_L2);
    matcher.match(desc_result1, desc_result2, matches);
    if (matches.size() == 0) {
        ptr_describe->releaseDescriptor();
        ptr_detector->releaseDetector();
        ptr_transform->releaseTransform();
        return "No matches found";
    }

    sort(matches.begin(), matches.end(), [](DMatch a, DMatch b) { return a.distance < b.distance; });

    (*transform_parameters) = ptr_transform->getTransform(kps1, kps2, matches);
    cout << endl;
    cout << (*transform_parameters).size() << " transform parameters as output" << endl;
    ptr_describe->releaseDescriptor();
    ptr_detector->releaseDetector();
    ptr_transform->releaseTransform();
    return "Transform parameters extraction is done";
}