#include "get_functions.h"

#include "DetectInterface.h"
#include "FeaturesInterface.h"
#include "Transformations.h"
#include "argvParser.h"
#include "getBOW.h"

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
        return "Success";
    }
    else if (type == 5)
    {
        for (int i = 0; i < desc_result.rows; i++)
        {
            const float *p = desc_result.ptr<float>(i);
            (*resultF).emplace_back(p, p + desc_result.cols);
        }
        return "Success";
    }
}

static bool matchCompare(DMatch x, DMatch y)
{
    return x.distance < y.distance;
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
    IDetector* ptr_detector = ChooseDetector(detector.c_str());

    map<string, double> det_params, desc_params;

    parseClient(detector_parameters, &det_params);
    parseClient(descriptor_parameters, &desc_params);

    cout << endl << "Setting parameters for detector:" << endl;
    ptr_detector->setParameters(det_params);

    (*kps) = ptr_detector->getPoints(image);
    ptr_detector->releaseDetector();

    if ((*kps).size() == 0)
        return "Zero keypoints detected";

    cout << endl;
    IDescribe* ptr_describe = ChooseFeatures(descriptor.c_str());
    cout << endl << endl << "Setting parameters for descriptor:" << endl;
    ptr_describe->setParameters(desc_params);
    Mat desc_result;
    try {
        desc_result = ptr_describe->getFeatures(kps, image);
    }
    catch (cv::Exception)
    {
        ptr_describe->releaseDescriptor();
        return "Features couldn't be computed for this combination of descriptor+detector for this image";
    }
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
    Mat desc_result;
    try {
        desc_result = ptr_describe->getFeatures(&kps, image);
    }
    catch (cv::Exception)
    {
        ptr_describe->releaseDescriptor();
        return "Features couldn't be computed for this combination of descriptor+detector for this image";
    }

    ptr_describe->releaseDescriptor();
    return descToVec(resultF, resultU, desc_result);
}

string getMatches(Mat desc1, Mat desc2, vector<DMatch>* matches)
{
    BFMatcher matcher(NORM_L2);
    vector<DMatch> match12, match21;
    matcher.match(desc1, desc2, match12);
    matcher.match(desc2, desc1, match21);
    for( size_t i = 0; i < match12.size(); i++ )
    {
        DMatch forward = match12[i];
        DMatch backward = match21[forward.trainIdx];
        if( backward.trainIdx == forward.queryIdx )
            matches->push_back( forward );
    }
    if ((*matches).size() == 0)
        return "No matches found";

    sort(matches->begin(), matches->end(), matchCompare);
    return "Success";
}

string getMatchesByImg(string image1, string image2, string detector, string detector_parameters, string descriptor,
        string descriptor_parameters, vector<KeyPoint> *kps1, vector<KeyPoint> *kps2, vector<DMatch>* matches)
{
    IDetector* ptr_detector = ChooseDetector(detector.c_str());

    map<string, double> det_params, desc_params;

    parseClient(detector_parameters, &det_params);
    parseClient(descriptor_parameters, &desc_params);

    cout << endl << "Setting parameters for detector:" << endl;
    ptr_detector->setParameters(det_params);

    (*kps1) = ptr_detector->getPoints(image1);
    (*kps2) = ptr_detector->getPoints(image2);
    ptr_detector->releaseDetector();

    if ((*kps1).size() == 0)
        return "Zero keypoints detected for the first image";

    if ((*kps2).size() == 0)
        return "Zero keypoints detected for the second image";

    cout << endl;
    IDescribe* ptr_describe = ChooseFeatures(descriptor.c_str());
    cout << endl << endl << "Setting parameters for descriptor:" << endl;
    ptr_describe->setParameters(desc_params);
    Mat desc_result1, desc_result2;
    try {
        desc_result1 = ptr_describe->getFeatures(kps1, image1);
    }
    catch (cv::Exception)
    {
        ptr_describe->releaseDescriptor();
        return "Features couldn't be computed for this combination of descriptor+detector for the first image";
    }

    try {
        desc_result2 = ptr_describe->getFeatures(kps2, image2);
    }
    catch (cv::Exception)
    {
        ptr_describe->releaseDescriptor();
        return "Features couldn't be computed for this combination of descriptor+detector for the second image";
    }

    ptr_describe->releaseDescriptor();

    return getMatches(desc_result1, desc_result2, matches);
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
    return "Success";
}

string getTransformParamsByImg(string image1, string image2, string detector, string detector_parameters,
                               string descriptor, string descriptor_parameters, string transformType, string transform_input_parameters,
                               vector<double> * transform_parameters)
{
    IDetector* ptr_detector = ChooseDetector(detector.c_str());
    map<string, double> det_params, desc_params, transf_params;
    vector<DMatch> matches;
    vector<KeyPoint> kps1, kps2;

    parseClient(detector_parameters, &det_params);
    parseClient(descriptor_parameters, &desc_params);
    parseClient(transform_input_parameters, &transf_params);

    cout << endl << "Setting parameters for detector:" << endl;
    ptr_detector->setParameters(det_params);

    kps1 = ptr_detector->getPoints(image1);
    kps2 = ptr_detector->getPoints(image2);
    ptr_detector->releaseDetector();
    if (kps1.size() == 0)
        return "Zero keypoints detected for the first image";

    if (kps2.size() == 0)
        return "Zero keypoints detected for the second image";
    cout << endl;
    IDescribe* ptr_describe = ChooseFeatures(descriptor.c_str());
    cout << endl << endl << "Setting parameters for descriptor:" << endl;
    ptr_describe->setParameters(desc_params);
    Mat desc_result1, desc_result2;

    try {
        desc_result1 = ptr_describe->getFeatures(&kps1, image1);
    }
    catch (cv::Exception)
    {
        ptr_describe->releaseDescriptor();
        return "Features couldn't be computed for this combination of descriptor+detector for the first image";
    }

    try {
        desc_result2 = ptr_describe->getFeatures(&kps2, image2);
    }
    catch (cv::Exception)
    {
        ptr_describe->releaseDescriptor();
        return "Features couldn't be computed for this combination of descriptor+detector for the second image";
    }
    ptr_describe->releaseDescriptor();
    getMatches(desc_result1, desc_result2, &matches);
    if (matches.size() == 0)
        return "No matches found";
    cout << endl;
    ITransform* ptr_transform = ChooseTransform(transformType.c_str());
    cout << endl << "Setting parameters for transform:" << endl;
    ptr_transform->setParameters(transf_params);
    (*transform_parameters) = ptr_transform->getTransform(kps1, kps2, matches);
    cout << endl;
    cout << (*transform_parameters).size() << " transform parameters as output" << endl;
    ptr_transform->releaseTransform();
    return "Success";
}

string getClosestImg(string q_image, vector<string>& imageBase, string descriptor, string descriptor_parameters,
                     string detector, string detector_parameters, int numOfClusters, int numOfImagesToRetrieve, vector<string>* retrievedImages,
                     vector<float> * distances)
{
    IDetector* ptr_detector = ChooseDetector(detector.c_str());
    IDescribe* ptr_describe = ChooseFeatures(descriptor.c_str());
    map<string, double> det_params, desc_params;
    parseClient(detector_parameters, &det_params);
    parseClient(descriptor_parameters, &desc_params);
    ptr_detector->setParameters(det_params);
    ptr_describe->setParameters(desc_params);

    vector<Mat> vectorBaseDescs;
    vector<int> idxsToDel;
    for (int i = 0; i < imageBase.size(); i++)
    {
        vector<KeyPoint> kps = ptr_detector->getPoints(imageBase[i]);
        if (kps.size() == 0)
            idxsToDel.push_back(i);
        else {
            Mat desc_result;
            try {
                desc_result = ptr_describe->getFeatures(&kps, imageBase[i]);
            }
            catch (cv::Exception)
            {
                idxsToDel.push_back(i);
                continue;
            }
            vectorBaseDescs.push_back(desc_result);
        }
    }
    for (int i = idxsToDel.size()-1; i > -1; i--)
    {
        imageBase.erase(imageBase.begin()+idxsToDel[i]);
    }
    
    if (imageBase.size() == 0)
    {
        ptr_describe->releaseDescriptor();
        ptr_detector->releaseDetector();
        return "No keypoints detected on input database. Check input detector, descriptor and parameters for them";
    }
    
    vector<fBow> processedBase;
    fbow::Vocabulary vocabulary;
    getBowVoc(vectorBaseDescs, &vocabulary, &processedBase);

    vector<KeyPoint> q_kps = ptr_detector->getPoints(q_image);
    if (q_kps.size() == 0)
    {
        ptr_describe->releaseDescriptor();
        ptr_detector->releaseDetector();
        return "No keypoints detected on query image";
    }
    Mat q_desc;
    try {
        q_desc = ptr_describe->getFeatures(&q_kps, q_image);
    }
    catch (cv::Exception)
    {
        return "Features can't be computed for this combination detector+descriptor for query image";
    }

    vector<pair<double, int>> matches = processOneDesc(processedBase, q_desc, vocabulary, numOfImagesToRetrieve);


    for (auto &oneMatch : matches) {
        string buf = imageBase[oneMatch.second];
        (*retrievedImages).push_back(buf);
        (*distances).push_back(oneMatch.first);
    }
    ptr_describe->releaseDescriptor();
    ptr_detector->releaseDetector();
    return "Success";
}