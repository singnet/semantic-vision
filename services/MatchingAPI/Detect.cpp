#include "DetectInterface.h"
#include <limits>

#include "Python.h"
#include "numpy/arrayobject.h"

static Mat getMat(string imageBytes)
{
    size_t length = imageBytes.size();
    Mat imageMat;
    vector<char> data((char *) imageBytes.c_str(), (char *) imageBytes.c_str() + length);
    imageMat = imdecode(data, IMREAD_UNCHANGED);
    return imageMat;
}


IDetector* ChooseDetector(const char *name)
{
    if (strncmp(name, (char*)"ORB", 3) == 0)
    {
        cout << "Using ORB detector" << endl;
        return ORBDetector::create();
    }
    else if (strncmp(name, (char*)"KAZE", 4) == 0)
    {
        cout << "Using KAZE detector" << endl;
        return KAZEDetector::create();
    }
    else if (strncmp(name, (char*)"AKAZE", 5) == 0)
    {
        cout << "Using AKAZE detector" << endl;
        return AKAZEDetector::create();
    }
    else if (strncmp(name, (char*)"AGAST", 5) == 0)
    {
        cout << "Using AGAST detector" << endl;
        return AGASTDetector::create();
    }
    else if (strncmp(name, (char*)"GFT", 3) == 0)
    {
        cout << "Using GFT detector" << endl;
        return GFTDetector::create();
    }
    else if (strncmp(name, (char*)"MSER", 4) == 0)
    {
        cout << "Using MSER detector" << endl;
        return MSERDetector::create();
    }
    else if (strncmp(name, (char*)"BRISK", 5) == 0)
    {
        cout << "Using BRISK detector" << endl;
        return BRISKDetector::create();
    }
    else if (strncmp(name, (char*)"FAST", 4) == 0)
    {
        cout << "Using FAST detector" << endl;
        return FASTDetector::create();
    }
    else if (strncmp(name, (char*)"BLOB", 4) == 0)
    {
        cout << "Using BLOB detector" << endl;
        return BLOBDetector::create();
    }
    else if (strncmp(name, (char*)"STAR", 4) == 0)
    {
        cout << "Using STAR detector" << endl;
        return STARDetector::create();
    }
    else if (strncmp(name, (char*)"MSDS", 4) == 0)
    {
        cout << "Using MSDS detector" << endl;
        return MSDSDetector::create();
    }
    else if (strncmp(name, (char*)"HLFD", 4) == 0)
    {
        cout << "Using HLFD detector" << endl;
        return HLFDDetector::create();
    }
    else if (strncmp(name, (char*)"Superpoint", 10) == 0)
    {
        cout << "Using Superpoint detector" << endl;
        return SPDetector::create();
    }
    else if (strncmp(name, (char*)"Magicpoint", 10) == 0)
    {
        cout << "Using Magicpoint detector" << endl;
        return MPDetector::create();
    }
    else
    {
        cout << "No detector was chosen or there is a mistake in the name. Using ORB by default" << endl;
        return ORBDetector::create();
    }
}

ORBDetector::ORBDetector()
{
    nfeatures = 500;
    scaleFactor = 1.2f;
    nlevels = 8;
    edgeThreshold = 31;
    firstLevel = 0;
}

void ORBDetector::setParameters(map<string, double> params)
{
    int imax = std::numeric_limits<int>::max();

    CheckParam_no_ifin<int> (params, (char*)"nfeatures", &nfeatures, 0, imax);

    CheckParam_no_ifin<float> (params, (char*)"scaleFactor", &scaleFactor, 1., 2.);

    CheckParam_no_ifin<int> (params, (char*)"nlevels", &nlevels, 1, 8);

    CheckParam_no_ifin<int> (params, (char*)"edgeThreshold", &edgeThreshold, 0, 255);

    CheckParam_no_ifin<int> (params, (char*)"firstLevel", &firstLevel, 0, 0);
}

vector<KeyPoint> ORBDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<ORB> featureDetector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, 2, ORB::HARRIS_SCORE, 31, 20);
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

ORBDetector* ORBDetector::create()
{
    auto* ptr_detector = new ORBDetector();
    return ptr_detector;
}

void ORBDetector::releaseDetector() {delete this;}

KAZEDetector::KAZEDetector()
{
    threshold = 0.001f;
    nOctaves = 4;
    nOctaveLayers = 4;
    diffusivity = KAZE::DIFF_PM_G2;
    diffTypes[0] = (char*)"DIFF_PM_G1";
    diffTypes[1] = (char*)"DIFF_PM_G2";
    diffTypes[2] = (char*)"DIFF_WEICKERT";
    diffTypes[3] = (char*)"DIFF_CHARBONNIER";
}

void KAZEDetector::setParameters(map<string, double> params)
{
    CheckParam_no_ifin<float> (params, (char*)"threshold", &threshold, 0., 1.);

    CheckParam_no_ifin<int> (params, (char*)"nOctaves", &nOctaves, 1, 20);

    CheckParam_no_ifin<int> (params, (char*)"nOctaveLayers", &nOctaveLayers, 1, 20);

    int range[] = {0, 1, 2, 3};
    CheckParam<int> (params, (char*)"diffusivity", &diffusivity, range, 4);
    cout << " - " << diffTypes[diffusivity];
}

vector<KeyPoint> KAZEDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<KAZE> featureDetector = KAZE::create(false, false, threshold, nOctaves, nOctaveLayers, (KAZE::DiffusivityType)diffusivity);
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

KAZEDetector* KAZEDetector::create()
{
    auto* ptr_detector = new KAZEDetector();
    return ptr_detector;
}

void KAZEDetector::releaseDetector() {delete this;}

AKAZEDetector::AKAZEDetector()
{
    threshold = 0.001f;
    nOctaves = 4;
    nOctaveLayers = 4;
    diffusivity = KAZE::DIFF_PM_G2;
    diffTypes[0] = (char*)"DIFF_PM_G1";
    diffTypes[1] = (char*)"DIFF_PM_G2";
    diffTypes[2] = (char*)"DIFF_WEICKERT";
    diffTypes[3] = (char*)"DIFF_CHARBONNIER";
}

void AKAZEDetector::setParameters(map<string, double> params)
{
    CheckParam_no_ifin<float> (params, (char*)"threshold", &threshold, 0., 1.);

    CheckParam_no_ifin<int> (params, (char*)"nOctaves", &nOctaves, 1, 20);

    CheckParam_no_ifin<int> (params, (char*)"nOctaveLayers", &nOctaveLayers, 1, 20);

    int range[] = {0, 1, 2, 3};
    CheckParam<int> (params, (char*)"diffusivity", &diffusivity, range, 4);
    cout << " - " << diffTypes[diffusivity];
}

vector<KeyPoint> AKAZEDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<AKAZE> featureDetector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3,  threshold, nOctaves, nOctaveLayers, (KAZE::DiffusivityType)diffusivity);
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

AKAZEDetector* AKAZEDetector::create()
{
    auto* ptr_detector = new AKAZEDetector();
    return ptr_detector;
}

void AKAZEDetector::releaseDetector() {delete this;}

AGASTDetector::AGASTDetector()
{
    threshold = 10;
    nonmaxSuppression = true;
    type = AgastFeatureDetector::OAST_9_16;
    types[0] = (char*)"AGAST_5_8";
    types[1] = (char*)"AGAST_7_12d";
    types[2] = (char*)"AGAST_7_12s";
    types[3] = (char*)"OAST_9_16";
}

void AGASTDetector::setParameters(map<string, double> params)
{
    CheckParam_no_ifin<int> (params, (char*)"threshold", &threshold, 0, 255);

    CheckParam_no_ifin<bool> (params, (char*)"nonmaxSuppression", &nonmaxSuppression, false, true);

    int range[] = {0, 1, 2, 3};
    CheckParam<int> (params, (char*)"type", &type, range, 4);

    cout << " - " << types[type];
}

vector<KeyPoint> AGASTDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<AgastFeatureDetector> featureDetector = AgastFeatureDetector::create(threshold, nonmaxSuppression, (AgastFeatureDetector::DetectorType)type);
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

AGASTDetector* AGASTDetector::create()
{
    auto* ptr_detector = new AGASTDetector();
    return ptr_detector;
}

void AGASTDetector::releaseDetector() {delete this;}

GFTDetector::GFTDetector()
{
    maxCorners = 1000;
    qualityLevel = 0.01;
    minDistance = 1;
    blockSize = 3;
    useHarrisDetector = false;
    k = 0.04;
}

void GFTDetector::setParameters(map<string, double> params)
{
    int imax = std::numeric_limits<int>::max();

    CheckParam_no_ifin<int> (params, (char*)"maxCorners", &maxCorners, 0, imax);

    CheckParam_no_ifin<double> (params, (char*)"qualityLevel", &qualityLevel, 0., 10000.);

    CheckParam_no_ifin<double> (params, (char*)"minDistance", &minDistance, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"blockSize", &blockSize, 0, imax);

    CheckParam_no_ifin<bool> (params, (char*)"useHarrisDetector", &useHarrisDetector, false, true);

    CheckParam_no_ifin<double> (params, (char*)"k", &k, 0, imax);
}

vector<KeyPoint> GFTDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<GFTTDetector> featureDetector = GFTTDetector::create(maxCorners, qualityLevel, minDistance, blockSize,
                                                             useHarrisDetector,
                                                             k );
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

GFTDetector* GFTDetector::create()
{
    auto* ptr_detector = new GFTDetector();
    return ptr_detector;
}

void GFTDetector::releaseDetector() {delete this;}

MSERDetector::MSERDetector()
{
    _delta = 5;
    _min_area = 60;
    _max_area = 14400;
    _max_variation = 0.25;
    _min_diversity = .2;
    _max_evolution = 200;
    _area_threshold = 1.01;
    _min_margin = 0.003;
    _edge_blur_size = 5;
}

void MSERDetector::setParameters(map<string, double> params)
{
    int imax = std::numeric_limits<int>::max();

    CheckParam_no_ifin<int> (params, (char*)"_delta", &_delta, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"_min_area", &_min_area, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"_max_area", &_max_area, 0, imax);

    CheckParam_no_ifin<double> (params, (char*)"_max_variation", &_max_variation, 0, imax);

    CheckParam_no_ifin<double> (params, (char*)"_min_diversity", &_min_diversity, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"_max_evolution", &_max_evolution, 0, imax);

    CheckParam_no_ifin<double> (params, (char*)"_area_threshold", &_area_threshold, 0, imax);

    CheckParam_no_ifin<double> (params, (char*)"_min_margin", &_min_margin, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"_edge_blur_size", &_edge_blur_size, 0, 9);
}

vector<KeyPoint> MSERDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<MSER> featureDetector = MSER::create(_delta,_min_area,_max_area,_max_variation,_min_diversity,_max_evolution,
                                             _area_threshold,_min_margin,_edge_blur_size );
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

MSERDetector* MSERDetector::create()
{
    auto* ptr_detector = new MSERDetector();
    return ptr_detector;
}

void MSERDetector::releaseDetector() {delete this;}

BRISKDetector::BRISKDetector()
{
    thresh = 30;
    octaves = 3;
}

void BRISKDetector::setParameters(map<string, double> params)
{
    CheckParam_no_ifin<int> (params, (char*)"thresh", &thresh, 0, 255);

    CheckParam_no_ifin<int> (params, (char*)"octaves", &octaves, 0, 9);
}

vector<KeyPoint> BRISKDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<BRISK> featureDetector = BRISK::create(thresh, octaves);
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

BRISKDetector* BRISKDetector::create()
{
    auto* ptr_detector = new BRISKDetector();
    return ptr_detector;
}

void BRISKDetector::releaseDetector() {delete this;}

FASTDetector::FASTDetector()
{
    threshold = 10;
    nonmaxSuppression = true;
    type = FastFeatureDetector::TYPE_9_16;
    types[0] = (char*)"TYPE_5_8";
    types[1] = (char*)"TYPE_7_12";
    types[2] = (char*)"TYPE_9_16";
}

void FASTDetector::setParameters(map<string, double> params)
{
    CheckParam_no_ifin<int> (params, (char*)"threshold", &threshold, 0, 255);

    CheckParam_no_ifin<bool> (params, (char*)"nonmaxSuppression", &nonmaxSuppression, false, true);

    int range[] = {0, 1, 2};
    CheckParam<int> (params, (char*)"type", &type, range, 3);
    cout << " - " << types[type];
}

vector<KeyPoint> FASTDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<FastFeatureDetector> featureDetector = FastFeatureDetector::create(threshold, nonmaxSuppression, (FastFeatureDetector::DetectorType)type);
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

FASTDetector* FASTDetector::create()
{
    auto* ptr_detector = new FASTDetector();
    return ptr_detector;
}

void FASTDetector::releaseDetector() {delete this;}

BLOBDetector::BLOBDetector() = default;

void BLOBDetector::setParameters(map<string, double> params)
{
    cout << endl << "No parameters needed for this detector" << endl;
}

vector<KeyPoint> BLOBDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<SimpleBlobDetector> featureDetector = SimpleBlobDetector::create();
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

BLOBDetector* BLOBDetector::create()
{
    auto* ptr_detector = new BLOBDetector();
    return ptr_detector;
}

void BLOBDetector::releaseDetector() {delete this;}

STARDetector::STARDetector()
{
    maxSize = 45;
    responseThreshold = 30;
    lineThresholdProjected = 10;
    lineThresholdBinarized = 8;
    suppressNonmaxSize = 5;
}

void STARDetector::setParameters(map<string, double> params)
{
    int imax = std::numeric_limits<int>::max();

    CheckParam_no_ifin<int> (params, (char*)"maxSize", &maxSize, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"responseThreshold", &responseThreshold, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"lineThresholdProjected", &lineThresholdProjected, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"lineThresholdBinarized", &lineThresholdBinarized, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"suppressNonmaxSize", &suppressNonmaxSize, 0, 39);
}

vector<KeyPoint> STARDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<xfeatures2d::StarDetector> featureDetector = xfeatures2d::StarDetector::create(maxSize,responseThreshold,lineThresholdProjected,
                                                                                       lineThresholdBinarized, suppressNonmaxSize);
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

STARDetector* STARDetector::create()
{
    auto* ptr_detector = new STARDetector();
    return ptr_detector;
}

void STARDetector::releaseDetector() {delete this;}

MSDSDetector::MSDSDetector()
{
    m_patch_radius = 3;
    m_search_area_radius = 5;
    m_nms_radius = 5;
    m_nms_scale_radius = 0;
    m_th_saliency = 250.0f;
    m_kNN = 4;
    m_scale_factor = 1.25f;
    m_n_scales = -1;
    m_compute_orientation = false;
}

void MSDSDetector::setParameters(map<string, double> params)
{
    int imax = std::numeric_limits<int>::max();

    CheckParam_no_ifin<int> (params, (char*)"m_patch_radius", &m_patch_radius, 0, 100);

    CheckParam_no_ifin<int> (params, (char*)"m_search_area_radius", &m_search_area_radius, 1, 100);

    CheckParam_no_ifin<int> (params, (char*)"m_nms_radius", &m_nms_radius, 0, imax);

    CheckParam_no_ifin<int> (params, (char*)"m_nms_scale_radius", &m_nms_scale_radius, 0, 1);

    CheckParam_no_ifin<float> (params, (char*)"m_th_saliency", &m_th_saliency, -imax, imax);

    CheckParam_no_ifin<int> (params, (char*)"m_kNN", &m_kNN, 1, 1000);

    CheckParam_no_ifin<float> (params, (char*)"m_scale_factor", &m_scale_factor, 1.01, 11.);

    int range[] = {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CheckParam<int> (params, (char*)"m_n_scales", &m_n_scales, range, 11);

    CheckParam_no_ifin<bool> (params, (char*)"m_compute_orientation", &m_compute_orientation, false, true);
}

vector<KeyPoint> MSDSDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<xfeatures2d::MSDDetector> featureDetector = xfeatures2d::MSDDetector::create(m_patch_radius,m_search_area_radius,m_nms_radius,
                                                                                     m_nms_scale_radius,m_th_saliency,m_kNN,m_scale_factor,m_n_scales,m_compute_orientation);
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

MSDSDetector* MSDSDetector::create()
{
    auto* ptr_detector = new MSDSDetector();
    return ptr_detector;
}

void MSDSDetector::releaseDetector() {delete this;}

HLFDDetector::HLFDDetector()
{
    numOctaves = 6;
    corn_thresh = 0.01f;
    DOG_thresh = 0.01f;
    maxCorners = 5000;
    num_layers = 4;
}

void HLFDDetector::setParameters(map<string, double> params)
{
    int imax = std::numeric_limits<int>::max();

    CheckParam_no_ifin<int> (params, (char*)"numOctaves", &numOctaves, 1, imax);

    CheckParam_no_ifin<float> (params, (char*)"corn_thresh", &corn_thresh, -imax, imax);

    CheckParam_no_ifin<float> (params, (char*)"DOG_thresh", &DOG_thresh, -imax, imax);

    CheckParam_no_ifin<int> (params, (char*)"maxCorners", &maxCorners, -imax, imax);

    int range[] = {2, 4};
    CheckParam<int> (params, (char*)"num_layers", &num_layers, range, 2);
}

vector<KeyPoint> HLFDDetector::getPoints(string image)
{
    Mat imageMat = getMat(image);
    vector<KeyPoint> keypoints;
    Ptr<xfeatures2d::HarrisLaplaceFeatureDetector> featureDetector = xfeatures2d::HarrisLaplaceFeatureDetector::create(numOctaves,
                                                                                                                       corn_thresh, DOG_thresh, maxCorners, num_layers);
    featureDetector->detect(imageMat, keypoints);
    return keypoints;
}

HLFDDetector* HLFDDetector::create()
{
    auto* ptr_detector = new HLFDDetector();
    return ptr_detector;
}

void HLFDDetector::releaseDetector() {delete this;}

MPDetector::MPDetector()
{
    threshold = 0;
}

void MPDetector::setParameters(map<string, double> params)
{
    float fmax = std::numeric_limits<float>::max();
    CheckParam_no_ifin<float> (params, (char*)"threshold", &threshold, 1, fmax);
}

vector<KeyPoint> MPDetector::getPoints(string image)
{
    const char* pyName = "getSuperPointKPs";

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* pName = PyUnicode_FromString(pyName);
    PyObject* pModule = PyImport_Import(pName);
    PyObject* pFunc = PyObject_GetAttrString(pModule, "getMagicPointKps");
    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyBytes_FromStringAndSize(image.c_str(), Py_ssize_t(image.size())));
    PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble(threshold));
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
    long* x = (long*)PyArray_DATA(PyTuple_GetItem(pValue, 1));
    long* y = (long*)PyArray_DATA(PyTuple_GetItem(pValue, 0));
    int length = PyArray_DIM(PyTuple_GetItem(pValue, 1), 0);

    vector<KeyPoint> result;
    for (int i = 0; i < length; i++)
    {
        KeyPoint buf;
        buf.pt.x = (int)x[i];
        buf.pt.y = (int)y[i];
        result.push_back(buf);
    }
    Py_DECREF(pArgs);
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(pValue);
    PyGILState_Release(gstate);
    return result;
}

MPDetector* MPDetector::create()
{
    auto* ptr_detector = new MPDetector();
    return ptr_detector;
}

void MPDetector::releaseDetector() {delete this;}

SPDetector::SPDetector()
{
    threshold = 0;
}

void SPDetector::setParameters(map<string, double> params)
{
    float fmax = std::numeric_limits<float>::max();
    CheckParam_no_ifin<float> (params, (char*)"threshold", &threshold, 1, fmax);
}

vector<KeyPoint> SPDetector::getPoints(string image)
{
    const char* pyName = "getSuperPointKPs";

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* pName = PyUnicode_FromString(pyName);
    PyObject* pModule = PyImport_Import(pName);
    PyObject* pFunc = PyObject_GetAttrString(pModule, "getSuperPointKps");
    cout << "1" << endl;
    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyBytes_FromStringAndSize(image.c_str(), Py_ssize_t(image.size())));
    cout << "2" << endl;
    PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble(threshold));
    cout << "3" << endl;
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
    cout << "4" << endl;
    long* x = (long*)PyArray_DATA(PyTuple_GetItem(pValue, 1));
    cout << "5" << endl;
    long* y = (long*)PyArray_DATA(PyTuple_GetItem(pValue, 0));
    cout << "6" << endl;
    int length = PyArray_DIM(PyTuple_GetItem(pValue, 1), 0);
    cout << "7" << endl;

    vector<KeyPoint> result;
    for (int i = 0; i < length; i++)
    {
        KeyPoint buf;
        buf.pt.x = (int)x[i];
        buf.pt.y = (int)y[i];
        result.push_back(buf);
    }

    Py_DECREF(pArgs);
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(pValue);
    PyGILState_Release(gstate);
    return result;
}

SPDetector* SPDetector::create()
{
    auto* ptr_detector = new SPDetector();
    return ptr_detector;
}

void SPDetector::releaseDetector() {delete this;}
